/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vins_common/ros_utils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

#include "estimator.h"
#include "solver_timeout.h"
#include "../utility/visualization.h"

namespace
{
struct CeresParameterSnapshot
{
    std::array<double, (WINDOW_SIZE + 1) * SIZE_POSE> pose;
    std::array<double, (WINDOW_SIZE + 1) * SIZE_SPEEDBIAS> speed_bias;
    std::array<double, NUM_OF_F * SIZE_FEATURE> feature;
    std::array<double, 2 * SIZE_POSE> ex_pose;
    std::array<double, 1> td;
};

// Frame period of the Schur-backend starvation summary log (see
// Estimator::schurReportStarvation). Chosen to match the existing
// stage_timing_interval default of 100 frames.
constexpr int kSchurStarvationReportFrames = 100;
}

Estimator::Estimator(): f_manager{Rs}
{
    RCLCPP_INFO(vins::logger(), "init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    stopping_ = true;
    measurements_ready_.notify_all();
    if (processThread.joinable()) processThread.join();
    clearState();
}


void Estimator::clearState()
{
    std::lock_guard<std::mutex> process_lock(mProcess);
    std::lock_guard<std::mutex> propagate_lock(mPropagate);
    latest_state_valid_ = false;
    latest_schur_active_ = false;
    latest_schur_state_ = vins::backend::EkfState();
    latest_imu_dt_ = 0.0;
    latest_input_time_ = -1.0;
    schur_last_imu_dt_ = 0.0;
    schur_last_imu_time_ = -1.0;
    {
        std::lock_guard<std::mutex> lock(mBuf);
        while (!accBuf.empty()) accBuf.pop();
        while (!gyrBuf.empty()) gyrBuf.pop();
        while (!featureBuf.empty()) featureBuf.pop();
    }

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    restart_requested_ = false;
    suppress_state_publication_ = false;

    schur_ekf_active_ = false;
    schur_frame_index_ = 0;
    schur_oldest_frame_ = 0;
    schur_update_count_ = 0;
    schur_stage_total_ms_ = 0.0;
    schur_features_total_ = 0.0;
    schur_rejected_total_ = 0.0;
    schur_soft_restart_time_ = -1.0;
    schur_starvation_monitor_.reset(0.0);
    schur_history_.clear();
    schur_cameras_.clear();
}

bool Estimator::restartAfterSolverFailure()
{
    if (!restart_requested_)
        return false;

    restart_requested_ = false;
    RCLCPP_WARN(vins::logger(),
        "Resetting estimator state after solver failure; current-frame observations and IMU "
        "preintegration inputs were already consumed, so the next frame starts a clean "
        "initialization instead of extending a stale image-frame history");
    clearState();
    setParameter();
    return true;
}

bool Estimator::runOptimization()
{
    return optimization();
}

void Estimator::maintainInitializationHistoryAfterFailure()
{
    slideWindow();

    // MARGIN_SECOND_NEW does not evict all_image_frame entries, and a
    // MARGIN_OLD slide can leave the discarded oldest frame behind. Keep only
    // headers that belong to the active window so the next initialization pass
    // cannot index pose/velocity arrays using stale image frames.
    trimInitializationHistoryToActiveWindow();

    const size_t max_history = static_cast<size_t>(frame_count + 1);
    if (all_image_frame.size() > max_history)
    {
        RCLCPP_WARN(vins::logger(),
            "Initialization image history still has %zu frames after trimming (limit %zu); "
            "scheduling a controlled estimator reset",
            all_image_frame.size(), max_history);
        scheduleControlledRestart();
    }
}

void Estimator::trimInitializationHistoryToActiveWindow()
{
    std::set<double> active_headers;
    for (int i = 0; i <= frame_count; ++i)
        active_headers.insert(Headers[i]);

    for (auto frame_it = all_image_frame.begin(); frame_it != all_image_frame.end();)
    {
        auto next_it = std::next(frame_it);
        if (active_headers.find(frame_it->first) == active_headers.end())
        {
            delete frame_it->second.pre_integration;
            frame_it->second.pre_integration = nullptr;
            all_image_frame.erase(frame_it);
        }
        frame_it = next_it;
    }
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    // Tracking remains full-rate; only backend observation packaging is gated.
    const bool submit = !MULTIPLE_THREAD || backend_gate_.accept(t);
    const bool draw = SHOW_TRACK && trackImageRequested();
    auto featureFrame = featureTracker.trackImage(t, _img, _img1, submit, draw);
    if (draw) pubTrackImage(featureTracker.getTrackImage(), t);
    if (!submit) return;
    {
        std::lock_guard<std::mutex> lock(mBuf);
        if (featureBuf.size() >= static_cast<size_t>(MAX_FEATURE_QUEUE)) {
            featureBuf.pop();
            if (++dropped_features_ == 1 || dropped_features_ % 100 == 0)
                RCLCPP_WARN(vins::logger(), "Backend overloaded: dropped %zu queued feature frames", dropped_features_);
        }
        featureBuf.emplace(t, std::move(featureFrame));
    }
    measurements_ready_.notify_one();
    if (!MULTIPLE_THREAD) processMeasurements();

}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    if (!std::isfinite(t) || !linearAcceleration.allFinite() || !angularVelocity.allFinite())
        return;
    // Serialize enqueue + prediction with visual re-anchoring. Otherwise a
    // sample can be replayed by updateLatestStates and then integrated twice
    // by the callback that originally enqueued it.
    std::lock_guard<std::mutex> propagate_lock(mPropagate);
    if (t <= latest_input_time_) return;
    // Measurement noise uses the sensor period, independent of the shorter
    // integration step after an image-time re-anchor or IMU tail replay.
    latest_imu_dt_ = latest_input_time_ >= 0.0 ? t - latest_input_time_ : 0.0;
    latest_input_time_ = t;
    {
        std::lock_guard<std::mutex> buffer_lock(mBuf);
        accBuf.emplace(t, linearAcceleration);
        gyrBuf.emplace(t, angularVelocity);
    }
    measurements_ready_.notify_one();

    if (latest_state_valid_ && fastPredictIMU(t, linearAcceleration, angularVelocity))
    {
        if (latest_schur_active_)
            pubLatestOdometry(latest_schur_state_, angularVelocity, latest_imu_dt_);
        else
            pubLatestOdometry(latest_P, latest_Q, latest_V, latest_time);
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    {
        std::lock_guard<std::mutex> lock(mBuf);
        if (featureBuf.size() >= static_cast<size_t>(MAX_FEATURE_QUEUE)) featureBuf.pop();
        featureBuf.emplace(t, featureFrame);
    }
    measurements_ready_.notify_one();
    if (!MULTIPLE_THREAD) processMeasurements();
}


bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    std::lock_guard<std::mutex> lock(mBuf);
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (!stopping_ && rclcpp::ok())
    {
        decltype(featureBuf)::value_type feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        {
            std::unique_lock<std::mutex> lock(mBuf);
            auto ready = [this] {
                return !featureBuf.empty() && (!USE_IMU ||
                    (!accBuf.empty() && featureBuf.front().first + td <= accBuf.back().first));
            };
            if (MULTIPLE_THREAD)
                measurements_ready_.wait(lock, [&] { return stopping_ || !rclcpp::ok() || ready(); });
            else if (!ready())
                return;
            if (stopping_ || !rclcpp::ok()) return;
            feature = std::move(featureBuf.front());
            featureBuf.pop();
            curTime = feature.first + td;
            if (USE_IMU) getIMUInterval(prevTime, curTime, accVector, gyrVector);
        }
        TicToc process_time;
        if (USE_IMU)
        {
            if (!initFirstPoseFlag) initFirstIMUPose(accVector);
            for (size_t i = 0; i < accVector.size(); ++i)
            {
                double dt;
                if (i == 0) dt = accVector[i].first - prevTime;
                else if (i == accVector.size() - 1) dt = curTime - accVector[i - 1].first;
                else dt = accVector[i].first - accVector[i - 1].first;
                processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
            }
        }
        bool restart_after_failure = false;
        bool suppress_publication = false;
        {
            std::lock_guard<std::mutex> lock(mProcess);
            processImage(feature.second, feature.first);
            restart_after_failure = restart_requested_;
            suppress_publication = suppress_state_publication_;
            if (!restart_after_failure)
            {
                // Frame inputs were consumed even when the solve failed, so the
                // IMU timeline must still advance on a retryable failure.
                prevTime = curTime;
            }
            if (!restart_after_failure && !suppress_publication)
            {
                printStatistics(*this, process_time.toc());
                std_msgs::msg::Header header;
                header.frame_id = "world";
                header.stamp = vins::stamp(feature.first);
                if (!rclcpp::ok()) return;
                pubOdometry(*this, header);
                pubKeyPoses(*this, header);
                pubCameraPose(*this, header);
                pubPointCloud(*this, header);
                pubKeyframe(*this);
                pubTF(*this, header);
            }
        }
        if (restart_after_failure)
        {
            // clearState()/setParameter() lock mProcess themselves, so the
            // reset must happen after releasing the processImage lock.
            restartAfterSolverFailure();
            if (!MULTIPLE_THREAD) break;
            continue;
        }
        if (!MULTIPLE_THREAD) break;
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;

    // The gyro rate variance uses the sensor sampling interval, not the
    // possibly fractional interval ending exactly at an image timestamp.
    if (schur_last_imu_time_ >= 0.0 && t > schur_last_imu_time_)
        schur_last_imu_dt_ = t - schur_last_imu_time_;
    schur_last_imu_time_ = t;
    if (schur_ekf_active_)
        schurPropagate(dt, linear_acceleration, angular_velocity);
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    RCLCPP_DEBUG(vins::logger(), "new image coming ------------------------------------------");
    RCLCPP_DEBUG(vins::logger(), "Adding feature points %lu", image.size());
    suppress_state_publication_ = false;
    const auto log_initialization_failure = [this](const char *stage)
    {
        suppress_state_publication_ = true;
        RCLCPP_WARN(vins::logger(),
            "Solver failed during %s; Ceres parameter blocks were rolled back, and "
            "double2vector, outlier rejection and state publication were skipped. The "
            "marginalization prior was not replaced. The initialization window is slid to keep "
            "the image-frame history bounded before the next retry; current-frame "
            "feature/image/IMU inputs were already advanced and are not rollback-capable",
            stage);
    };
    const auto log_odometry_failure = [this](const char *stage)
    {
        suppress_state_publication_ = true;
        RCLCPP_WARN(vins::logger(),
            "Solver failed during %s; Ceres parameter blocks were rolled back, and "
            "double2vector, outlier rejection, state publication and window sliding were skipped. "
            "The marginalization prior was not replaced. Current-frame feature/image/IMU inputs "
            "were already advanced and are not rollback-capable, so a controlled estimator reset "
            "is scheduled before the next frame",
            stage);
        scheduleControlledRestart();
    };
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    RCLCPP_DEBUG(vins::logger(), "%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    RCLCPP_DEBUG(vins::logger(), "Solving %d", frame_count);
    RCLCPP_DEBUG(vins::logger(), "number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // These frame-level inputs are consumed before the solver runs. They are
    // intentionally not part of the solver rollback: a failed solve may leave
    // the current feature/image/IMU observations advanced, but it must not
    // replace the marginalization prior. INITIAL retries still apply the
    // window-maintenance slide below so all_image_frame stays bounded;
    // NON_LINEAR failures skip sliding and request a controlled reset.
    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        RCLCPP_INFO(vins::logger(), "calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                RCLCPP_WARN(vins::logger(), "initial extrinsic rotation calib success");
                RCLCPP_WARN_STREAM(vins::logger(), "initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    if (!runOptimization())
                    {
                        log_initialization_failure("monocular IMU initialization");
                        maintainInitializationHistoryAfterFailure();
                        return;
                    }
                    solver_flag = NON_LINEAR;
                    activateSchurBackend(image, header);
                    updateLatestStates();
                    slideWindow();
                    RCLCPP_INFO(vins::logger(), "Initialization finish!");
                }
                else if (restart_requested_)
                {
                    // initialStructure() already found an inconsistent history.
                    // Do not touch the window again; processMeasurements() will
                    // perform the controlled reset after releasing mProcess.
                    suppress_state_publication_ = true;
                    RCLCPP_WARN(vins::logger(),
                        "Initialization requested a controlled reset; skipping window sliding "
                        "for the current frame");
                    return;
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                if (all_image_frame.size() != static_cast<size_t>(frame_count + 1))
                {
                    suppress_state_publication_ = true;
                    RCLCPP_WARN(vins::logger(),
                        "Stereo IMU initialization image history has %zu frames for an active "
                        "window of %d; scheduling a controlled estimator reset",
                        all_image_frame.size(), frame_count + 1);
                    scheduleControlledRestart();
                    return;
                }
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    if (i > frame_count ||
                        (i > 0 && frame_it->second.pre_integration == nullptr))
                    {
                        suppress_state_publication_ = true;
                        RCLCPP_WARN(vins::logger(),
                            "Stereo IMU initialization has an invalid image frame at window "
                            "index %d; scheduling a controlled estimator reset", i);
                        scheduleControlledRestart();
                        return;
                    }
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                if (!runOptimization())
                {
                    log_initialization_failure("stereo IMU initialization");
                    maintainInitializationHistoryAfterFailure();
                    return;
                }
                solver_flag = NON_LINEAR;
                activateSchurBackend(image, header);
                updateLatestStates();
                slideWindow();
                RCLCPP_INFO(vins::logger(), "Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            const bool initial_solve_ok = runOptimization();
            if (!initial_solve_ok)
                log_initialization_failure("stereo-only initialization");

            if(frame_count == WINDOW_SIZE)
            {
                if (!initial_solve_ok)
                {
                    maintainInitializationHistoryAfterFailure();
                    return;
                }
                if (!runOptimization())
                {
                    log_initialization_failure("stereo-only initialization refinement");
                    maintainInitializationHistoryAfterFailure();
                    return;
                }
                solver_flag = NON_LINEAR;
                activateSchurBackend(image, header);
                updateLatestStates();
                slideWindow();
                RCLCPP_INFO(vins::logger(), "Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        bool solve_ok = false;
        if (schur_ekf_active_)
        {
            // ARM-oriented path: propagate + Schur-complement EKF update, no
            // Ceres solve and no marginalization factor.
            solve_ok = schurProcessFrame(image, header);
            if (!solve_ok)
            {
                RCLCPP_WARN(vins::logger(),
                    "Schur EKF could not process this frame; re-seeding the filter from a "
                    "Ceres solve to keep the estimate valid");
                solve_ok = runOptimization();
                if (solve_ok)
                    initSchurBackend();
            }
        }
        else
        {
            solve_ok = runOptimization();
        }
        if (!solve_ok)
        {
            log_odometry_failure("nonlinear odometry");
            return;
        }
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }
            
        RCLCPP_DEBUG(vins::logger(), "solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            suppress_state_publication_ = true;
            RCLCPP_WARN(vins::logger(),
                "Failure detection triggered; scheduling a controlled estimator reset");
            failure_occur = 1;
            scheduleControlledRestart();
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // Optional Ceres warm-up: start the EKF from a converged window state
        // once the configured number of post-initialisation frames has passed.
        if (!schur_ekf_active_ && BACKEND_TYPE == "schur_ekf")
            activateSchurBackend(image, header);
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
        if (std::getenv("VINS_SCHUR_DEBUG") != nullptr)
        {
            static int ceres_dbg = 0;
            if (++ceres_dbg % 5 == 0)
                fprintf(stderr,
                        "[ceres-state] frame=%d t=%.3f |p|=%.4f |v|=%.4f |ba|=%.5f |bg|=%.6f "
                        "ba=[%.4f %.4f %.4f] bg=[%.5f %.5f %.5f]\n",
                        ceres_dbg, header + td, Ps[WINDOW_SIZE].norm(), Vs[WINDOW_SIZE].norm(),
                        Bas[WINDOW_SIZE].norm(), Bgs[WINDOW_SIZE].norm(), Bas[WINDOW_SIZE].x(),
                        Bas[WINDOW_SIZE].y(), Bas[WINDOW_SIZE].z(), Bgs[WINDOW_SIZE].x(),
                        Bgs[WINDOW_SIZE].y(), Bgs[WINDOW_SIZE].z());
        }
    }  
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    if (frame_count < 0 || frame_count > WINDOW_SIZE || all_image_frame.empty())
    {
        RCLCPP_WARN(vins::logger(),
            "Initialization has an invalid frame history (frame_count=%d, image_frames=%zu); "
            "scheduling a controlled estimator reset",
            frame_count, all_image_frame.size());
        scheduleControlledRestart();
        return false;
    }
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //RCLCPP_WARN(vins::logger(), "IMU variation %f!", var);
        if(var < 0.25)
        {
            RCLCPP_INFO(vins::logger(), "IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        RCLCPP_INFO(vins::logger(), "Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        RCLCPP_DEBUG(vins::logger(), "global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        if (i > frame_count)
        {
            RCLCPP_WARN(vins::logger(),
                "Initialization image history has frames beyond the active window "
                "(index=%d, frame_count=%d); scheduling a controlled estimator reset",
                i, frame_count);
            scheduleControlledRestart();
            return false;
        }
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
            if (i > frame_count)
            {
                RCLCPP_WARN(vins::logger(),
                    "Initialization frame timestamps do not match the active window; "
                    "scheduling a controlled estimator reset");
                scheduleControlledRestart();
                return false;
            }
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            RCLCPP_DEBUG(vins::logger(), "Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            RCLCPP_DEBUG(vins::logger(), "solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        RCLCPP_INFO(vins::logger(), "misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        RCLCPP_DEBUG(vins::logger(), "solve g failed!");
        return false;
    }

    // Old image frames can outlive the active window when initialization was
    // retried. Clear stale keyframe flags first so the velocity slot below can
    // never be indexed past the active window.
    for (auto &frame_entry : all_image_frame)
        frame_entry.second.is_key_frame = false;

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        auto frame_entry = all_image_frame.find(Headers[i]);
        if (frame_entry == all_image_frame.end())
        {
            RCLCPP_WARN(vins::logger(),
                "Initialization window header %d is missing from all_image_frame; "
                "scheduling a controlled estimator reset",
                i);
            scheduleControlledRestart();
            return false;
        }
        Matrix3d Ri = frame_entry->second.R;
        Vector3d Pi = frame_entry->second.T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        frame_entry->second.is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            if (kv > frame_count || kv > WINDOW_SIZE)
            {
                RCLCPP_WARN(vins::logger(),
                    "Initialization velocity index %d exceeds the active window (frame_count=%d); "
                    "scheduling a controlled estimator reset",
                    kv, frame_count);
                scheduleControlledRestart();
                return false;
            }
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    if (kv != frame_count)
    {
        RCLCPP_WARN(vins::logger(),
            "Initialization found %d keyframes for %d active frames; scheduling a controlled "
            "estimator reset",
            kv + 1, frame_count + 1);
        scheduleControlledRestart();
        return false;
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    RCLCPP_DEBUG_STREAM(vins::logger(), "g0     " << g.transpose());
    RCLCPP_DEBUG_STREAM(vins::logger(), "my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                RCLCPP_DEBUG(vins::logger(), "average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            RCLCPP_DEBUG(vins::logger(), "euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        RCLCPP_INFO(vins::logger(), " little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        RCLCPP_INFO(vins::logger(), " big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        RCLCPP_INFO(vins::logger(), " big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        RCLCPP_INFO(vins::logger(), " big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //RCLCPP_INFO(vins::logger(), " big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //RCLCPP_INFO(vins::logger(), " big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        RCLCPP_INFO(vins::logger(), " big delta_angle ");
        //return true;
    }
    return false;
}

// ================= Schur-complement EKF backend (ARM-oriented) =============
//
// Opt-in replacement for the Ceres sliding-window solve, selected with
// `backend_type: schur_ekf`.  The window is a 15-state IMU block plus 6 states
// per retained clone; feature points are nuisance parameters that never enter
// the covariance (Schur-eliminated per track).  See docs/BACKEND_SCHUR_EKF.md.

void Estimator::initSchurBackend()
{
    vins::backend::EkfOptions options;
    options.window_size = SCHUR_WINDOW_SIZE;
    options.gravity = g.norm() > 1e-3 ? g.norm() : G.norm();
    // Continuous-time spectral densities (unit^2/Hz).  ACC_N/GYR_N/ACC_W/GYR_W
    // follow VINS's per-sample convention (they enter IntegrationBase through a
    // V matrix scaled by 0.5*dt) and must not be fed to the EKF as densities:
    // doing so inflates the velocity/attitude process noise by ~1/dt and makes
    // every visual update over-correct the velocity.
    options.acc_noise = SCHUR_ACC_NOISE;
    options.gyr_noise = SCHUR_GYR_NOISE;
    options.acc_bias_noise = SCHUR_ACC_BIAS_NOISE;
    options.gyr_bias_noise = SCHUR_GYR_BIAS_NOISE;
    options.init_position_sigma = SCHUR_INIT_POS_SIGMA;
    options.init_velocity_sigma = SCHUR_INIT_VEL_SIGMA;
    options.init_rotation_sigma = SCHUR_INIT_ROT_SIGMA;
    options.init_acc_bias_sigma = SCHUR_INIT_BA_SIGMA;
    options.init_gyr_bias_sigma = SCHUR_INIT_BG_SIGMA;

    const Eigen::Quaterniond q_init(Rs[frame_count]);
    const Eigen::Vector4d q_vec(q_init.w(), q_init.x(), q_init.y(), q_init.z());
    // Sensor time (frame header + td) of the seeding frame. It is the baseline
    // of the time-based starvation criterion until the first applied update.
    const double seed_time = Headers[frame_count] + td;
    schur_state_.reset(options, seed_time, q_vec, Ps[frame_count],
                       Vs[frame_count], Bas[frame_count], Bgs[frame_count]);

    vins::backend::UpdateOptions update_options;
    update_options.max_features = SCHUR_MAX_FEATURES;
    update_options.min_observations = SCHUR_MIN_OBSERVATIONS;
    update_options.pixel_noise = SCHUR_PIXEL_NOISE;
    update_options.focal_length = FOCAL_LENGTH;
    update_options.chi2_multiplier = SCHUR_CHI2_MULTIPLIER;
    update_options.triangulation_iterations =
        static_cast<int>(std::lround(SCHUR_TRIANGULATION_ITERATIONS));
    update_options.min_features = SCHUR_MIN_FEATURES;
    update_options.min_update_observations = SCHUR_MIN_UPDATE_OBSERVATIONS;
    update_options.info_rank_rtol = SCHUR_INFO_RANK_RTOL;
    update_options.info_indefinite_tol = SCHUR_INFO_INDEFINITE_TOL;
    update_options.point_block_rtol = SCHUR_POINT_BLOCK_RTOL;
    update_options.condition_limit = SCHUR_CONDITION_LIMIT;
    schur_updater_ = vins::backend::SchurUpdater(update_options);

    schur_cameras_.clear();
    for (int i = 0; i < std::max(NUM_OF_CAM, 1); ++i)
    {
        vins::backend::CameraExtrinsics camera;
        if (i < NUM_OF_CAM)
        {
            const Eigen::Quaterniond q_cb(ric[i]);
            camera.q_cb = Eigen::Vector4d(q_cb.w(), q_cb.x(), q_cb.y(), q_cb.z());
            camera.p_cb = tic[i];
        }
        schur_cameras_.push_back(camera);
    }

    schur_history_.clear();
    schur_frame_index_ = 0;
    schur_oldest_frame_ = 0;
    schur_update_count_ = 0;
    schur_stage_total_ms_ = 0.0;
    schur_features_total_ = 0.0;
    schur_rejected_total_ = 0.0;
    schur_lost_total_ = 0.0;
    schur_marginalized_total_ = 0.0;
    schur_consecutive_failures_ = 0;
    schur_skipped_frames_ = 0;
    // A full (re-)seed ends any pending soft-restart episode.
    schur_soft_restart_time_ = -1.0;
    // Time-based starvation policy: configure the thresholds and re-seed the
    // monitor so the seed frame counts as the last applied update. The
    // diagnostic counters start fresh for the new filter instance; the restart
    // cooldown anchor deliberately survives reset() (see StarvationMonitor), so
    // a controlled restart cannot be followed by another one within
    // SCHUR_RESTART_COOLDOWN_SECONDS of sensor time.
    vins::backend::StarvationMonitor::Config starvation_config;
    starvation_config.max_no_update_seconds = SCHUR_MAX_NO_UPDATE_SECONDS;
    starvation_config.starvation_seconds = SCHUR_STARVATION_SECONDS;
    starvation_config.noise_scale = SCHUR_STARVATION_NOISE_SCALE;
    starvation_config.deep_starvation_seconds = SCHUR_STARVATION_DEEP_SECONDS;
    starvation_config.deep_noise_scale = SCHUR_STARVATION_DEEP_NOISE_SCALE;
    starvation_config.restart_cooldown_seconds = SCHUR_RESTART_COOLDOWN_SECONDS;
    schur_starvation_monitor_.configure(starvation_config);
    schur_starvation_monitor_.reset(seed_time);
    // P7 hold policy: configure and drop any holding state from a previous
    // filter instance (cumulative diagnostics survive).
    vins::backend::StarvationHold::Config hold_config;
    hold_config.enabled = SCHUR_STARVATION_HOLD;
    hold_config.recovery_min_features = SCHUR_MIN_FEATURES;
    hold_config.recovery_frames = SCHUR_HOLD_RECOVERY_FRAMES;
    schur_hold_.configure(hold_config);
    schur_hold_.deactivate();
    schur_rebuild_frames_ = 0;
    schur_rebuild_pending_ = false;
    schur_ekf_active_ = true;

    RCLCPP_INFO(vins::logger(),
                "Schur EKF seeded at t=%.6f: p=[%.3f %.3f %.3f] v=[%.3f %.3f %.3f] "
                "|g|=%.4f window=%d",
                seed_time, Ps[frame_count].x(), Ps[frame_count].y(),
                Ps[frame_count].z(), Vs[frame_count].x(), Vs[frame_count].y(),
                Vs[frame_count].z(), options.gravity, options.window_size);
}

void Estimator::activateSchurBackend(
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    if (BACKEND_TYPE != "schur_ekf")
        return;
    if (!USE_IMU)
    {
        RCLCPP_WARN(vins::logger(),
                    "backend_type=schur_ekf requires an IMU; keeping the Ceres backend");
        return;
    }
    if (schur_ekf_active_)
        return;
    if (inputImageCnt < SCHUR_WARMUP_FRAMES)
        return;

    initSchurBackend();
    if (!schurProcessFrame(image, header))
    {
        schur_ekf_active_ = false;
        RCLCPP_WARN(vins::logger(),
                    "Schur EKF could not consume the initialisation frame; keeping the "
                    "Ceres backend");
        return;
    }
    RCLCPP_INFO(vins::logger(),
                "Schur-complement EKF backend active (window=%d clones, max_features=%d)",
                SCHUR_WINDOW_SIZE, SCHUR_MAX_FEATURES);
}

void Estimator::schurPropagate(double dt, const Eigen::Vector3d &acc,
                               const Eigen::Vector3d &gyr)
{
    if (!schur_ekf_active_)
        return;
    // Guard against clock jumps and long IMU gaps: a single step that large
    // would be dominated by linearisation error.
    if (!(dt > 0.0) || dt > 1.0)
        return;
    // Visual-starvation process-noise inflation (P3/P4). The monitor returns the
    // multiplier of the current starvation stage: 1.0 when not starved,
    // SCHUR_STARVATION_NOISE_SCALE while starved, and
    // SCHUR_STARVATION_DEEP_NOISE_SCALE once deep starvation
    // (SCHUR_STARVATION_DEEP_SECONDS) is reached. The stage is evaluated once
    // per image frame in schurProcessFrame(), where the sensor-time gap since the
    // last applied update is known; between frames it is piecewise constant, so
    // every IMU step of a starved stretch uses the same scale. A successful
    // update clears the stage (scale back to 1.0) in the same frame.
    // P7 hold: the hold policy must let P grow naturally instead of being
    // re-inflated, so the starvation Q multiplier is bypassed while holding.
    const double noise_scale =
        schur_hold_.active() ? 1.0 : schur_starvation_monitor_.noise_scale();
    schur_state_.propagate(acc, gyr, dt, noise_scale);
}

bool Estimator::schurProcessFrame(
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    if (!schur_ekf_active_)
        return false;

    TicToc t_update;
    // Sensor time of this frame (frame header + td). The time-based starvation
    // criterion compares it against the baseline held by schur_starvation_monitor_
    // (the time of the last *applied* update, or the seed time right after
    // initSchurBackend()).
    const double frame_time = header + td;
    // P6 starvation-adaptive weak update: the scale for *this* frame comes from
    // the current starvation gap (time since the last applied visual update).
    // 1.0 while the gap is at or below SCHUR_STARVATION_SECONDS, ramping to
    // SCHUR_WEAK_UPDATE_MAX_SCALE at SCHUR_STARVATION_DEEP_SECONDS. Disabled or
    // invalid parameters give 1.0, which is a strict no-op.
    double weak_scale = vins::backend::weakUpdateScale(
        schur_starvation_monitor_.starvation_duration(frame_time),
        SCHUR_STARVATION_WEAK_UPDATE, SCHUR_STARVATION_SECONDS,
        SCHUR_STARVATION_DEEP_SECONDS, SCHUR_WEAK_UPDATE_MAX_SCALE);

    // Collect this frame's camera-0 observations once. Column layout is
    // [x, y, 1, u, v, vx, vy] where (x, y) are normalised coordinates. The hold
    // policy needs the count before it touches any visual structure.
    std::vector<std::pair<int, Eigen::Vector2d>> observations;
    observations.reserve(image.size());
    for (const auto &feature : image)
    {
        for (const auto &observation : feature.second)
        {
            if (observation.first != 0)
                continue;  // the Schur backend consumes camera 0 only
            const Eigen::Matrix<double, 7, 1> &value = observation.second;
            observations.emplace_back(feature.first, Eigen::Vector2d(value(0), value(1)));
        }
    }

    // P7 hold fast path. While holding, this frame is a pure-IMU frame: no
    // history entry, no clone augmentation, no visual update and no
    // marginalisation. The live state keeps propagating (schurPropagate()) and
    // is written back/published every frame, so odometry stays continuous. Once
    // the tracker reports enough observations for enough consecutive frames,
    // the stale clone blocks/history are dropped (live P untouched) and this
    // same frame starts rebuilding the window by falling through to the normal
    // path. That frame carries a single history entry, so no track can be
    // eligible and the stale observations are never absorbed.
    if (schur_hold_.active())
    {
        const vins::backend::StarvationHold::Action hold_action =
            schur_hold_.observe_frame(frame_time, static_cast<int>(observations.size()));
        if (hold_action == vins::backend::StarvationHold::Action::kHold)
        {
            if (!schurStateFinite())
            {
                schur_ekf_active_ = false;
                RCLCPP_WARN(vins::logger(),
                            "Schur EKF state became non-finite while holding at t=%.3f; "
                            "falling back to the Ceres backend", frame_time);
                return false;
            }
            if (schur_hold_.frames() % kSchurStarvationReportFrames == 0)
                schurReportStarvation(frame_time);
            ++schur_update_count_;
            schurWriteBackWindow();
            schurReportStageTiming(t_update.toc());
            return true;
        }
        schurHoldRecover(frame_time, static_cast<int>(observations.size()));
        // The monitor was re-anchored, so the weak-update ramp starts at zero.
        weak_scale = vins::backend::weakUpdateScale(
            schur_starvation_monitor_.starvation_duration(frame_time),
            SCHUR_STARVATION_WEAK_UPDATE, SCHUR_STARVATION_SECONDS,
            SCHUR_STARVATION_DEEP_SECONDS, SCHUR_WEAK_UPDATE_MAX_SCALE);
    }
    const bool debug_schur = std::getenv("VINS_SCHUR_DEBUG") != nullptr;
    auto covariance_bounds = [](const Eigen::MatrixXd &P, double &trace, double &min_eig,
                                double &max_eig) {
        trace = P.trace();
        min_eig = std::numeric_limits<double>::quiet_NaN();
        max_eig = std::numeric_limits<double>::quiet_NaN();
        if (!P.allFinite()) return;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(0.5 * (P + P.transpose()));
        if (eig.info() == Eigen::Success) {
            min_eig = eig.eigenvalues().minCoeff();
            max_eig = eig.eigenvalues().maxCoeff();
        }
    };
    double p_trace_before = 0.0, p_min_before = 0.0, p_max_before = 0.0;
    double p_trace_augmented = 0.0, p_min_augmented = 0.0, p_max_augmented = 0.0;
    if (debug_schur)
        covariance_bounds(schur_state_.covariance(), p_trace_before, p_min_before, p_max_before);
    const vins::backend::ImuNominal imu_before_update = schur_state_.imu();

    // 1. Record this frame's normalised, undistorted observations (collected
    //    above). A frame that just left hold only carries this single entry, so
    //    no track can pass the min-observation gate and the clone window is
    //    rebuilt frame by frame without absorbing stale tracks.
    schur_history_.add_frame(schur_frame_index_, 0, observations);

    // 2. Append the current pose as the newest clone. Pruning is deferred until
    //    after the visual update so the oldest clone can still be observed by
    //    the tracks that are about to be marginalised (OpenVINS/SchurVINS
    //    ordering: augment -> update -> marginalise).
    schur_state_.augment_clone(header + td);
    if (debug_schur)
        covariance_bounds(schur_state_.covariance(), p_trace_augmented, p_min_augmented,
                          p_max_augmented);

    // 3. Clone i corresponds to frame index (schur_oldest_frame_ + i), so the
    //    oldest retained clone is the one that will be pruned in step 6.
    schur_oldest_frame_ = schur_frame_index_ - (schur_state_.num_clones() - 1);
    schur_history_.trim(schur_oldest_frame_);

    // 4. MSCKF track selection. Only features that are lost in this frame, or
    //    whose oldest observation belongs to the clone that is about to be
    //    margined out, may enter the update. Each such track is consumed
    //    exactly once, which is what keeps the filter consistent: re-inserting
    //    the same rows on consecutive frames double counts the measurements,
    //    collapses the covariance and produces spurious velocity corrections.
    // OpenVINS/SchurVINS rule (VioManager.cpp:366-369): while the clone window
    // is still filling up there is nothing to marginalise, so only features
    // that are actually lost this frame may be used. Only once the window is
    // full (i.e. this frame will prune the oldest clone) do we also admit
    // tracks whose oldest observation sits on that outgoing clone.
    const bool window_full =
        schur_state_.num_clones() > schur_state_.options().window_size;
    vins::backend::TrackSelection selection = schur_history_.tracks(
        schur_oldest_frame_, schur_frame_index_,
        schur_updater_.options().min_observations,
        static_cast<size_t>(schur_updater_.options().max_features),
        window_full);

    vins::backend::UpdateSummary summary;
    // P6: inject the per-frame weak update scale into the updater options before
    // the update. With weak_scale == 1.0 this leaves the update unchanged.
    schur_updater_.set_weak_update_scale(weak_scale);
    if (!selection.tracks.empty())
    {
        summary = schur_updater_.update(schur_state_, selection.tracks, schur_cameras_);
        schur_features_total_ += summary.features_used;
        schur_rejected_total_ += summary.features_rejected;
    }
    schur_lost_total_ += selection.lost_count;
    schur_marginalized_total_ += selection.marginalized_count;

    // 5. Consume the tracks that actually entered the update. A rejected track
    //    is kept so it can be retried while it is still inside the window; the
    //    window trim bounds its lifetime.
    if (summary.applied)
    {
        schur_history_.erase_features(summary.used_feature_ids);
        schur_consecutive_failures_ = 0;
        // A successful update ends any starvation: it re-baselines the time
        // criterion and switches the process-noise inflation back off (the
        // inter-update gap is folded into the monitor's diagnostic maximum). It
        // also clears the P5 soft-restart escalation timer.
        schur_soft_restart_time_ = -1.0;
        schur_starvation_monitor_.note_applied_update(frame_time);
    }
    else
    {
        ++schur_consecutive_failures_;
        ++schur_skipped_frames_;
        if (schur_consecutive_failures_ == 1 || schur_consecutive_failures_ % 10 == 0)
            RCLCPP_WARN(vins::logger(),
                        "Schur EKF skipped frame %d (streak=%d, tracks=%zu lost=%d marg=%d, "
                        "used=%d rejected=%d, weak_scale=%.2f): %s",
                        schur_frame_index_, schur_consecutive_failures_, selection.tracks.size(),
                        selection.lost_count, selection.marginalized_count,
                        summary.features_used, summary.features_rejected, weak_scale,
                        summary.failure_reason.empty() ? "no eligible tracks"
                                                       : summary.failure_reason.c_str());

        // P6: a weak-update frame that is *still* rejected is the case the
        // feature exists to fix, so surface it (count-throttled: 1st and every
        // 20th occurrence). The counter is process-wide and diagnostic only.
        if (weak_scale > 1.0)
        {
            static int weak_skip_warn_count = 0;
            ++weak_skip_warn_count;
            if (weak_skip_warn_count == 1 || weak_skip_warn_count % 20 == 0)
                RCLCPP_WARN(vins::logger(),
                            "Schur EKF weak-update frame %d still skipped (occurrence=%d, "
                            "weak_scale=%.2f, tracks=%zu used=%d rejected=%d): %s",
                            schur_frame_index_, weak_skip_warn_count, weak_scale,
                            selection.tracks.size(), summary.features_used,
                            summary.features_rejected,
                            summary.failure_reason.empty()
                                ? "no eligible tracks"
                                : summary.failure_reason.c_str());
        }

        // Time-based starvation criterion (P1/P4). Semantics: evaluated only
        // here, i.e. once per image frame that did NOT apply a visual update; it
        // is never evaluated inside schurPropagate(), so a single large IMU gap
        // cannot trigger a restart by itself. A frame that DOES apply an update
        // re-baselines the monitor above and never reaches this code.
        // The monitor re-baselines on a backwards timestamp (bag replay / clock
        // rollback) instead of restarting, and drives the process-noise
        // inflation stage (P3/P4) from the gap.
        // A restart is a *last resort*: on ARM it costs a full Ceres
        // re-initialisation (10-44 s without odometry), so the time criterion is
        // the primary guard (SCHUR_MAX_NO_UPDATE_SECONDS, default 3 s, versus
        // starvation stretches that usually self-heal within 1-2 s) and
        // SCHUR_MAX_CONSECUTIVE_FAILURES is only a frame-count backstop
        // (default 60). Both are suppressed while the restart cooldown
        // (SCHUR_RESTART_COOLDOWN_SECONDS) is active, which bounds a restart
        // storm to at most one restart per cooldown window.
        const vins::backend::StarvationMonitor::Decision starvation =
            schur_starvation_monitor_.observe_failed_frame(frame_time);
        if (starvation.deep_escalated)
            RCLCPP_WARN(vins::logger(),
                        "Schur EKF deep starvation started at frame %d (%.3f s without a "
                        "visual update > %.3f s): escalating the process-noise scale to %.2f",
                        schur_frame_index_, starvation.gap_seconds,
                        SCHUR_STARVATION_DEEP_SECONDS, SCHUR_STARVATION_DEEP_NOISE_SCALE);
        bool restarted = false;
        // P5 soft starvation restart. When enabled while the Schur EKF is
        // active, the time-criterion restart keeps the IMU/EKF nominal state
        // instead of tearing the estimator down (see schurSoftRestart()).
        // Escalation backstop: a soft restart that is not followed by an applied
        // visual update within SCHUR_SOFT_RESTART_MAX_SECONDS of sensor time is
        // given up on and replaced by one full controlled restart. The anchor is
        // only armed by the *first* soft restart of an unrecovered episode (see
        // schurSoftRestart()), so a chain of repeat soft restarts cannot postpone
        // the escalation forever.
        const bool soft_mode = SCHUR_STARVATION_SOFT_RESTART && schur_ekf_active_;
        // P7 hold mode never restarts: the time criterion and the frame-count
        // backstop both enter the hold state instead. It takes precedence over
        // the P5 soft restart (and therefore over its escalation backstop).
        const bool hold_mode = SCHUR_STARVATION_HOLD && schur_ekf_active_;
        if (!hold_mode && soft_mode &&
            vins::backend::softRestartEscalationDue(
                frame_time, schur_soft_restart_time_, SCHUR_SOFT_RESTART_MAX_SECONDS))
        {
            RCLCPP_ERROR(vins::logger(),
                         "Schur EKF soft restart at t=%.3f did not apply a visual update "
                         "for %.3f s (> %.3f s) at frame %d; escalating to a full "
                         "controlled estimator restart",
                         schur_soft_restart_time_, frame_time - schur_soft_restart_time_,
                         SCHUR_SOFT_RESTART_MAX_SECONDS, schur_frame_index_);
            schur_soft_restart_time_ = -1.0;
            schur_consecutive_failures_ = 0;
            schur_starvation_monitor_.note_restart(frame_time);
            scheduleControlledRestart();
            restarted = true;
        }
        else if (starvation.restart)
        {
            if (hold_mode)
            {
                // "不重启、只保持": keep the live state and P, stop accepting
                // visual updates and keep publishing pure-IMU propagation.
                // schurHoldEnter() logs the gap and dropped clone count.
                schurHoldEnter(frame_time, starvation.gap_seconds);
            }
            else if (soft_mode)
            {
                // Keep propagating/publishing from the current IMU state; only
                // the visual structure is dropped. schurSoftRestart() logs the
                // gap, frame index and dropped clone count itself.
                schurSoftRestart(frame_time, starvation.gap_seconds);
            }
            else
            {
                RCLCPP_ERROR(vins::logger(),
                             "Schur EKF visually starved for %.3f s (> %.3f s) at frame %d: "
                             "streak=%d skipped_total=%d max_no_update_gap=%.3f s; scheduling "
                             "a controlled estimator restart",
                             starvation.gap_seconds, SCHUR_MAX_NO_UPDATE_SECONDS,
                             schur_frame_index_, schur_consecutive_failures_,
                             schur_skipped_frames_,
                             schur_starvation_monitor_.max_gap_seconds());
                schur_consecutive_failures_ = 0;
                // Arm the cooldown before scheduling: the restart runs
                // clearState()/initSchurBackend(), which re-seed the monitor via
                // reset() and must not drop the anchor.
                schur_starvation_monitor_.note_restart(frame_time);
                scheduleControlledRestart();
            }
            restarted = true;
        }
        else if (starvation.cooldown_suppressed)
        {
            if (hold_mode)
            {
                // The cooldown only bounds restarts; entering hold is free, so a
                // cooldown-suppressed trigger still enters hold.
                schurHoldEnter(frame_time, starvation.gap_seconds);
                restarted = true;
            }
            else
            {
                const int suppressed = schur_starvation_monitor_.cooldown_suppressions();
                if (suppressed == 1 || suppressed % 10 == 0)
                    RCLCPP_WARN(vins::logger(),
                                "Schur EKF restart suppressed by the cooldown (%.3f s without a "
                                "visual update, %.3f s of %.3f s cooldown elapsed) at frame %d: "
                                "suppressed_total=%d streak=%d skipped_total=%d",
                                starvation.gap_seconds,
                                frame_time - schur_starvation_monitor_.last_restart_time(),
                                SCHUR_RESTART_COOLDOWN_SECONDS, schur_frame_index_, suppressed,
                                schur_consecutive_failures_, schur_skipped_frames_);
            }
        }
        // Frame-count backstop: only when the time criterion did not restart this
        // frame AND the cooldown is not holding restarts back. The failure streak
        // keeps accumulating while suppressed so the diagnostic is not lost. In
        // P5 soft mode the backstop also takes the soft path (keep the IMU/EKF
        // state) instead of clearState(); only the escalation backstop is allowed
        // to schedule a full restart in that mode.
        const vins::backend::StarvationRestartAction backstop_action =
            vins::backend::frameCountBackstopAction(
                restarted, schur_starvation_monitor_.cooldown_active(frame_time),
                schur_consecutive_failures_, SCHUR_MAX_CONSECUTIVE_FAILURES, soft_mode,
                hold_mode);
        if (backstop_action == vins::backend::StarvationRestartAction::kHold)
        {
            RCLCPP_ERROR(vins::logger(),
                         "Schur EKF produced %d consecutive frames without a visual update "
                         "(%d skipped in total) at frame %d; entering hold "
                         "(no restart, keeping the IMU/EKF state and covariance)",
                         schur_consecutive_failures_, schur_skipped_frames_,
                         schur_frame_index_);
            schurHoldEnter(frame_time, starvation.gap_seconds);
        }
        else if (backstop_action == vins::backend::StarvationRestartAction::kSoft)
        {
            RCLCPP_ERROR(vins::logger(),
                         "Schur EKF produced %d consecutive frames without a visual update "
                         "(%d skipped in total) at frame %d; performing a soft restart "
                         "(keeping the IMU/EKF state, no re-initialisation)",
                         schur_consecutive_failures_, schur_skipped_frames_,
                         schur_frame_index_);
            schurSoftRestart(frame_time, starvation.gap_seconds);
        }
        else if (backstop_action == vins::backend::StarvationRestartAction::kFull)
        {
            RCLCPP_ERROR(vins::logger(),
                         "Schur EKF produced %d consecutive frames without a visual update "
                         "(%d skipped in total); scheduling a controlled estimator restart",
                         schur_consecutive_failures_, schur_skipped_frames_);
            schur_consecutive_failures_ = 0;
            schur_starvation_monitor_.note_restart(frame_time);
            scheduleControlledRestart();
        }
    }

    // 6. Drop the oldest clone now that the tracks observing it have been used.
    //    A frame that just entered hold has no clones left, so skip the index
    //    bookkeeping (schurHoldRecover() re-seeds it on recovery).
    if (schur_state_.num_clones() > 0)
    {
        schur_state_.prune_clone_window();
        schur_oldest_frame_ = schur_frame_index_ - (schur_state_.num_clones() - 1);
        schur_history_.trim(schur_oldest_frame_);
    }

    // 6b. P7 rebuild accounting: after leaving hold the window is rebuilt one
    //     clone per frame. Report once it is back to full size so the recovery
    //     cost is visible in the logs.
    if (schur_rebuild_pending_)
    {
        ++schur_rebuild_frames_;
        if (schur_state_.num_clones() >= schur_state_.options().window_size)
        {
            schur_rebuild_pending_ = false;
            schur_rebuild_last_frames_ = schur_rebuild_frames_;
            RCLCPP_INFO(vins::logger(),
                        "Schur EKF clone window rebuilt after hold in %d frames "
                        "(%d clones, t=%.3f): visual updates resume from the next frame",
                        schur_rebuild_frames_, schur_state_.num_clones(), frame_time);
        }
    }

    if (debug_schur)
    {
        double p_trace_after = 0.0, p_min_after = 0.0, p_max_after = 0.0;
        covariance_bounds(schur_state_.covariance(), p_trace_after, p_min_after, p_max_after);
        const auto &im = schur_state_.imu();
        const Eigen::Vector3d dv = im.v - imu_before_update.v;
        const Eigen::Vector3d dba = im.ba - imu_before_update.ba;
        const Eigen::Vector3d dbg = im.bg - imu_before_update.bg;
        const Eigen::Vector3d dp = im.p - imu_before_update.p;
        fprintf(stderr,
                "[schur-step] frame=%d t=%.3f n=%d tracks=%zu lost=%d marg=%d used=%d rej=%d "
                "obs=%d dof=%d applied=%d rank=%d cond=%.3g reason='%s' "
                "P=[%.4g/%.3g/%.3g] -> aug=[%.4g/%.3g/%.3g] -> post=[%.4g/%.3g/%.3g] "
                "|dp|=%.3g |dv|=%.3g |dba|=%.3g |dbg|=%.3g "
                "p=%.4f v=%.4f ba=%.4f bg=%.5f rms=%.3f maxres=%.3f\n",
                schur_frame_index_, header + td, schur_state_.size(), selection.tracks.size(),
                selection.lost_count, selection.marginalized_count, summary.features_used,
                summary.features_rejected, summary.observations_used, summary.degrees_of_freedom,
                static_cast<int>(summary.applied), summary.rank, summary.condition,
                summary.failure_reason.c_str(), p_trace_before, p_min_before, p_max_before,
                p_trace_augmented, p_min_augmented, p_max_augmented, p_trace_after, p_min_after,
                p_max_after, dp.norm(), dv.norm(), dba.norm(), dbg.norm(), im.p.norm(),
                im.v.norm(), im.ba.norm(), im.bg.norm(), summary.mean_residual_px,
                summary.max_residual_px);
    }
    // Periodic starvation summary (every kSchurStarvationReportFrames frames).
    if (schur_frame_index_ > 0 &&
        schur_frame_index_ % kSchurStarvationReportFrames == 0)
        schurReportStarvation(frame_time);

    ++schur_frame_index_;
    ++schur_update_count_;

    if (!schurStateFinite())
    {
        schur_ekf_active_ = false;
        RCLCPP_WARN(vins::logger(),
                    "Schur EKF state became non-finite at frame %d; falling back to the "
                    "Ceres backend", schur_frame_index_ - 1);
        return false;
    }

    // 7. Publish the corrected state through the existing window arrays so the
    //    visualisation and odometry output paths stay unchanged.
    schurWriteBackWindow();
    schurReportStageTiming(t_update.toc());
    return true;
}

void Estimator::schurHoldEnter(double frame_time, double gap_seconds)
{
    if (!schur_hold_.enter(frame_time))
        return;
    const int frame_index = schur_frame_index_;
    // Drop only the visual structure. Every clone block is hard-deleted, so the
    // live 15-dim IMU block of P survives bit-for-bit: no reset(), no
    // re-inflation. This also keeps the per-IMU propagation cost at 15x15 while
    // holding instead of growing with a stale window.
    const int dropped_clones = schur_state_.drop_all_clones();
    schur_history_.clear();
    schur_frame_index_ = 0;
    schur_oldest_frame_ = 0;
    schur_consecutive_failures_ = 0;
    // Hold supersedes any pending P5 escalation: nothing is escalated while we
    // hold, because holding never restarts.
    schur_soft_restart_time_ = -1.0;
    RCLCPP_WARN(vins::logger(),
                "Schur EKF hold entered at frame %d (t=%.3f): %.3f s without a visual "
                "update (> %.3f s). No restart and no covariance re-inflation: dropped %d "
                "clones, live state and P kept; pure-IMU propagation continues and the "
                "estimate keeps being published. Episode=%d",
                frame_index, frame_time, gap_seconds, SCHUR_MAX_NO_UPDATE_SECONDS,
                dropped_clones, schur_hold_.episodes());
}

void Estimator::schurHoldRecover(double frame_time, int observations)
{
    const int episode = schur_hold_.episodes();
    const double duration = schur_hold_.leave(frame_time);
    // The tracker recovered: drop whatever visual structure is left and start a
    // fresh frame numbering. The live state and its covariance (which grew
    // naturally during the hold) are deliberately kept untouched.
    schur_state_.drop_all_clones();
    schur_history_.clear();
    schur_frame_index_ = 0;
    schur_oldest_frame_ = 0;
    schur_consecutive_failures_ = 0;
    // Re-anchor the time criterion so the rebuilt window has a full
    // SCHUR_MAX_NO_UPDATE_SECONDS grace period to fill before it can fire again.
    schur_starvation_monitor_.rebaseline(frame_time);
    schur_rebuild_frames_ = 0;
    schur_rebuild_pending_ = true;
    RCLCPP_WARN(vins::logger(),
                "Schur EKF hold exit after %.3f s (episode=%d, t=%.3f): tracker recovered "
                "with %d observations (>= %d for %d consecutive frames). Rebuilding the "
                "clone window from this frame, live state and P kept",
                duration, episode, frame_time, observations, SCHUR_MIN_FEATURES,
                SCHUR_HOLD_RECOVERY_FRAMES);
}

void Estimator::schurSoftRestart(double frame_time, double gap_seconds)
{
    // P5 soft starvation restart. Keep the IMU/EKF nominal state (q/p/v/ba/bg)
    // and drop only the visual structure: no clearState(), no Ceres/SfM
    // re-initialisation, no solver_flag change and no publication suppression,
    // so IMU propagation and odometry output continue through the starvation
    // stretch. This is what removes the 27-37 s odometry gap that a full
    // controlled restart costs on ARM.
    const vins::backend::ImuNominal imu = schur_state_.imu();
    const int dropped_clones = schur_state_.num_clones();
    const int frame_index = schur_frame_index_;
    // Discard the clone window and the old covariance, re-inflating P from the
    // configured init_*_sigma values while preserving the motion state.
    schur_state_.reset(schur_state_.options(), frame_time, imu.q, imu.p, imu.v, imu.ba,
                       imu.bg);
    schur_history_.clear();
    schur_frame_index_ = 0;
    schur_oldest_frame_ = 0;
    schur_consecutive_failures_ = 0;
    schur_skipped_frames_ = 0;
    // Arm the escalation timer only when no soft restart is already pending:
    // repeated soft restarts during one unrecovered episode must not postpone
    // the full-restart escalation (see softRestartEscalationDue()).
    if (schur_soft_restart_time_ < 0.0)
        schur_soft_restart_time_ = frame_time;
    // Same cooldown semantics as a full restart (P4).
    schur_starvation_monitor_.note_restart(frame_time);
    // schur_ekf_active_ is deliberately left true.
    RCLCPP_WARN(vins::logger(),
                "Schur EKF soft restart at frame %d: visually starved for %.3f s "
                "(> %.3f s); dropped %d clones, keeping the IMU/EKF state "
                "(q/p/v/ba/bg) without re-initialisation. A full restart is scheduled "
                "if no visual update is applied within %.3f s",
                frame_index, gap_seconds, SCHUR_MAX_NO_UPDATE_SECONDS, dropped_clones,
                SCHUR_SOFT_RESTART_MAX_SECONDS);
}

void Estimator::schurWriteBackWindow()
{
    const vins::backend::ImuNominal &imu = schur_state_.imu();
    const Eigen::Quaterniond q_imu(imu.q(0), imu.q(1), imu.q(2), imu.q(3));
    Rs[frame_count] = q_imu.toRotationMatrix();
    Ps[frame_count] = imu.p;
    Vs[frame_count] = imu.v;
    Bas[frame_count] = imu.ba;
    Bgs[frame_count] = imu.bg;

    // Refresh the retained clones into the matching window slots so the
    // visualiser shows the corrected (not the propagated) poses.
    const int num_clones = schur_state_.num_clones();
    for (int i = 0; i < num_clones; ++i)
    {
        const int window_index = frame_count - (num_clones - 1 - i);
        if (window_index < 0)
            continue;
        const vins::backend::CloneNominal &clone = schur_state_.clone(i);
        const Eigen::Quaterniond q_clone(clone.q(0), clone.q(1), clone.q(2), clone.q(3));
        Rs[window_index] = q_clone.toRotationMatrix();
        Ps[window_index] = clone.p;
    }
}

bool Estimator::schurStateFinite() const
{
    const vins::backend::ImuNominal &imu = schur_state_.imu();
    if (!imu.q.allFinite() || !imu.p.allFinite() || !imu.v.allFinite() ||
        !imu.ba.allFinite() || !imu.bg.allFinite())
        return false;
    if (!schur_state_.covariance().allFinite())
        return false;
    for (int i = 0; i < schur_state_.num_clones(); ++i)
    {
        const vins::backend::CloneNominal &clone = schur_state_.clone(i);
        if (!clone.q.allFinite() || !clone.p.allFinite())
            return false;
    }
    return true;
}

void Estimator::schurReportStageTiming(double update_ms)
{
    schur_stage_total_ms_ += update_ms;
    if (!STAGE_TIMING || STAGE_TIMING_INTERVAL < 1)
        return;
    if (schur_update_count_ % STAGE_TIMING_INTERVAL != 0)
        return;

    const double samples = static_cast<double>(STAGE_TIMING_INTERVAL);
    RCLCPP_INFO(vins::logger(),
                "schur_ekf stage timing (%d samples): update=%.3f ms, used=%.1f, "
                "rejected=%.1f, lost=%.1f, marg=%.1f, skipped=%d, state=%dx%d",
                STAGE_TIMING_INTERVAL, schur_stage_total_ms_ / samples,
                schur_features_total_ / samples, schur_rejected_total_ / samples,
                schur_lost_total_ / samples, schur_marginalized_total_ / samples,
                schur_skipped_frames_, schur_state_.size(), schur_state_.size());
    schur_stage_total_ms_ = 0.0;
    schur_features_total_ = 0.0;
    schur_rejected_total_ = 0.0;
    schur_lost_total_ = 0.0;
    schur_marginalized_total_ = 0.0;
}

void Estimator::schurReportStarvation(double frame_time) const
{
    // P6: report the weak-update scale this frame would use, recomputed from the
    // same starvation gap so the summary stays consistent with schurProcessFrame.
    const double weak_scale = vins::backend::weakUpdateScale(
        schur_starvation_monitor_.starvation_duration(frame_time),
        SCHUR_STARVATION_WEAK_UPDATE, SCHUR_STARVATION_SECONDS,
        SCHUR_STARVATION_DEEP_SECONDS, SCHUR_WEAK_UPDATE_MAX_SCALE);
    RCLCPP_INFO(vins::logger(),
                "schur_ekf starvation summary (%d frames): time_gate_triggers=%d "
                "deep_escalations=%d cooldown_suppressions=%d noise_scale=%.2f "
                "weak_update_scale=%.2f max_no_update_gap=%.3f s current_starvation=%.3f s "
                "consecutive_failures=%d skipped_total=%d "
                "thresholds(max_no_update=%.2f s cooldown=%.2f s) "
                "hold(enabled=%d active=%d episodes=%d frames=%d current=%.3f s "
                "total=%.3f s last=%.3f s max=%.3f s streak=%d/%d rebuild=%s "
                "rebuild_frames=%d)",
                schur_frame_index_, schur_starvation_monitor_.time_gate_triggers(),
                schur_starvation_monitor_.deep_escalations(),
                schur_starvation_monitor_.cooldown_suppressions(),
                schur_starvation_monitor_.noise_scale(), weak_scale,
                schur_starvation_monitor_.max_gap_seconds(),
                schur_starvation_monitor_.starvation_duration(frame_time),
                schur_consecutive_failures_, schur_skipped_frames_,
                SCHUR_MAX_NO_UPDATE_SECONDS, SCHUR_RESTART_COOLDOWN_SECONDS,
                static_cast<int>(SCHUR_STARVATION_HOLD),
                static_cast<int>(schur_hold_.active()), schur_hold_.episodes(),
                schur_hold_.frames(), schur_hold_.current_duration(frame_time),
                schur_hold_.total_duration(), schur_hold_.last_duration(),
                schur_hold_.max_duration(), schur_hold_.good_streak(),
                SCHUR_HOLD_RECOVERY_FRAMES, schur_rebuild_pending_ ? "pending" : "idle",
                schur_rebuild_last_frames_);
}

bool Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    const bool stage_timing_enabled = STAGE_TIMING;
    TicToc stage_timer;
    double stage_vector2double_ms = 0.0;
    double stage_problem_ms = 0.0;
    double stage_solve_ms = 0.0;
    double stage_double2vector_ms = 0.0;
    double stage_marginalization_ms = 0.0;
    static double stage_vector2double_total_ms = 0.0;
    static double stage_problem_total_ms = 0.0;
    static double stage_solve_total_ms = 0.0;
    static double stage_double2vector_total_ms = 0.0;
    static double stage_marginalization_total_ms = 0.0;
    static std::size_t stage_timing_samples = 0;
    auto finish_stage_timing = [&]()
    {
        if (!stage_timing_enabled)
            return;
        stage_vector2double_total_ms += stage_vector2double_ms;
        stage_problem_total_ms += stage_problem_ms;
        stage_solve_total_ms += stage_solve_ms;
        stage_double2vector_total_ms += stage_double2vector_ms;
        stage_marginalization_total_ms += stage_marginalization_ms;
        ++stage_timing_samples;
        if (stage_timing_samples >= static_cast<std::size_t>(STAGE_TIMING_INTERVAL))
        {
            const double samples = static_cast<double>(stage_timing_samples);
            RCLCPP_INFO(vins::logger(),
                "optimization stage timing (%zu samples): vector2double=%.3f ms, problem=%.3f ms, "
                "solve=%.3f ms, double2vector=%.3f ms, marginalization=%.3f ms",
                stage_timing_samples,
                stage_vector2double_total_ms / samples,
                stage_problem_total_ms / samples,
                stage_solve_total_ms / samples,
                stage_double2vector_total_ms / samples,
                stage_marginalization_total_ms / samples);
            stage_vector2double_total_ms = 0.0;
            stage_problem_total_ms = 0.0;
            stage_solve_total_ms = 0.0;
            stage_double2vector_total_ms = 0.0;
            stage_marginalization_total_ms = 0.0;
            stage_timing_samples = 0;
        }
    };

    if (stage_timing_enabled)
        stage_timer.tic();
    vector2double();
    if (stage_timing_enabled)
        stage_vector2double_ms = stage_timer.toc();

    if (stage_timing_enabled)
        stage_timer.tic();
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::Manifold *local_parameterization = new PoseManifold();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    const bool estimate_ex_pose =
        (ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::Manifold *local_parameterization = new PoseManifold();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!estimate_ex_pose)
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    RCLCPP_DEBUG(vins::logger(), "visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());
    if (stage_timing_enabled)
        stage_problem_ms = stage_timer.toc();

    const bool transactional_solver = (SOLVER_TIMEOUT_POLICY != "warn");
    CeresParameterSnapshot parameter_snapshot{};
    std::memcpy(parameter_snapshot.pose.data(), para_Pose, sizeof(para_Pose));
    std::memcpy(parameter_snapshot.speed_bias.data(), para_SpeedBias, sizeof(para_SpeedBias));
    std::memcpy(parameter_snapshot.feature.data(), para_Feature, sizeof(para_Feature));
    std::memcpy(parameter_snapshot.ex_pose.data(), para_Ex_Pose, sizeof(para_Ex_Pose));
    std::memcpy(parameter_snapshot.td.data(), para_Td, sizeof(para_Td));
    auto restore_parameter_snapshot = [&]()
    {
        std::memcpy(para_Pose, parameter_snapshot.pose.data(), sizeof(para_Pose));
        std::memcpy(para_SpeedBias, parameter_snapshot.speed_bias.data(), sizeof(para_SpeedBias));
        std::memcpy(para_Feature, parameter_snapshot.feature.data(), sizeof(para_Feature));
        std::memcpy(para_Ex_Pose, parameter_snapshot.ex_pose.data(), sizeof(para_Ex_Pose));
        std::memcpy(para_Td, parameter_snapshot.td.data(), sizeof(para_Td));
    };
    static std::size_t timeout_count = 0;
    static std::size_t retry_attempt_count = 0;
    static std::size_t retry_success_count = 0;
    static std::size_t revert_count = 0;
    bool reverted = false;
    auto rollback_parameters = [&]()
    {
        if (reverted)
            return;
        restore_parameter_snapshot();
        reverted = true;
        ++revert_count;
    };

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    const double first_solver_budget = options.max_solver_time_in_seconds;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    if (stage_timing_enabled)
        stage_timer.tic();
    ceres::Solve(options, &problem, &summary);
    if (stage_timing_enabled)
        stage_solve_ms += stage_timer.toc();
    const double first_solver_elapsed_s = t_solver.toc() / 1000.0;

    const bool timed_out = vins::solverTimedOut(summary, first_solver_elapsed_s, first_solver_budget);
    bool solved = summary.IsSolutionUsable();

    if (timed_out)
    {
        ++timeout_count;

        if (!transactional_solver)
        {
            static rclcpp::Clock throttle_clock(RCL_STEADY_TIME);
            if (solved)
            {
                RCLCPP_WARN_THROTTLE(vins::logger(), throttle_clock, 5000,
                    "Ceres hit max_solver_time=%.0f ms after %.1f ms and %zu iterations "
                    "(policy=warn); accepting the truncated but usable solution, which can bias "
                    "the marginalization prior",
                    first_solver_budget * 1000.0, first_solver_elapsed_s * 1000.0,
                    summary.iterations.size());
            }
            else
            {
                RCLCPP_WARN_THROTTLE(vins::logger(), throttle_clock, 5000,
                    "Ceres hit max_solver_time=%.0f ms after %.1f ms and %zu iterations "
                    "(policy=warn); the solution is unusable and will be rolled back",
                    first_solver_budget * 1000.0, first_solver_elapsed_s * 1000.0,
                    summary.iterations.size());
            }
        }
        else if (SOLVER_TIMEOUT_POLICY == "retry")
        {
            ++retry_attempt_count;
            restore_parameter_snapshot();

            const double retry_solver_budget = first_solver_budget * SOLVER_TIMEOUT_RETRY_MULTIPLIER;
            options.max_solver_time_in_seconds = retry_solver_budget;
            ceres::Solver::Summary retry_summary;
            TicToc retry_solver_timer;
            if (stage_timing_enabled)
                stage_timer.tic();
            ceres::Solve(options, &problem, &retry_summary);
            if (stage_timing_enabled)
                stage_solve_ms += stage_timer.toc();
            const double retry_solver_elapsed_s = retry_solver_timer.toc() / 1000.0;

            const bool retry_timed_out =
                vins::solverTimedOut(retry_summary, retry_solver_elapsed_s, retry_solver_budget);
            solved = !retry_timed_out && retry_summary.IsSolutionUsable();
            if (solved)
            {
                ++retry_success_count;
            }
            else
            {
                rollback_parameters();
            }

            RCLCPP_WARN(vins::logger(),
                "Ceres timeout (policy=retry): first solve hit max_solver_time=%.0f ms after "
                "%.1f ms and %zu iterations; retry with %.0f ms %s after %.1f ms and %zu "
                "iterations; solver parameter rollback=%s",
                first_solver_budget * 1000.0, first_solver_elapsed_s * 1000.0,
                summary.iterations.size(), retry_solver_budget * 1000.0,
                solved ? "succeeded" : "failed", retry_solver_elapsed_s * 1000.0,
                retry_summary.iterations.size(), reverted ? "true" : "false");
        }
        else
        {
            rollback_parameters();
            solved = false;
            RCLCPP_WARN(vins::logger(),
                "Ceres timeout (policy=%s): hit max_solver_time=%.0f ms after %.1f ms and "
                "%zu iterations; retry_attempted=false, solver parameter rollback=true. "
                "Marginalization prior and sliding window remain unchanged; current-frame "
                "feature/image/IMU inputs were already advanced and are not rolled back",
                SOLVER_TIMEOUT_POLICY.c_str(), first_solver_budget * 1000.0,
                first_solver_elapsed_s * 1000.0, summary.iterations.size());
        }

        if (timeout_count % 100 == 0)
        {
            RCLCPP_INFO(vins::logger(),
                "Ceres timeout cumulative summary (policy=%s): timeouts=%zu retry_attempts=%zu "
                "retry_successes=%zu reverts=%zu",
                SOLVER_TIMEOUT_POLICY.c_str(), timeout_count, retry_attempt_count,
                retry_success_count, revert_count);
        }
    }
    else if (!solved)
    {
        rollback_parameters();
        RCLCPP_WARN(vins::logger(),
            "Ceres solution unusable (policy=%s) after %.1f ms and %zu iterations with "
            "max_solver_time=%.0f ms; solver parameter rollback=true. Marginalization prior "
            "and sliding window remain unchanged; current-frame feature/image/IMU inputs "
            "were already advanced and are not rolled back",
            SOLVER_TIMEOUT_POLICY.c_str(), first_solver_elapsed_s * 1000.0,
            summary.iterations.size(), first_solver_budget * 1000.0);
    }

    //cout << summary.BriefReport() << endl;
    RCLCPP_DEBUG(vins::logger(), "Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    if (!solved)
    {
        rollback_parameters();
        RCLCPP_WARN(vins::logger(),
            "Optimization failed: reverted solver parameter blocks only; marginalization prior "
            "and sliding window were not advanced, and no state will be published. Current-frame "
            "feature/image/IMU inputs were already advanced and are not part of this rollback");
        finish_stage_timing();
        return false;
    }

    if (stage_timing_enabled)
        stage_timer.tic();
    double2vector();
    if (stage_timing_enabled)
        stage_double2vector_ms = stage_timer.toc();
    if (estimate_ex_pose)
        openExEstimation = 1;
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
    {
        finish_stage_timing();
        return true;
    }

    if (stage_timing_enabled)
        stage_timer.tic();
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        RCLCPP_DEBUG(vins::logger(), "pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        RCLCPP_DEBUG(vins::logger(), "marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            RCLCPP_DEBUG(vins::logger(), "begin marginalization");
            marginalization_info->preMarginalize();
            RCLCPP_DEBUG(vins::logger(), "end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            RCLCPP_DEBUG(vins::logger(), "begin marginalization");
            marginalization_info->marginalize();
            RCLCPP_DEBUG(vins::logger(), "end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    if (stage_timing_enabled)
        stage_marginalization_ms = stage_timer.toc();
    finish_stage_timing();
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
    return true;
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            if (all_image_frame.find(t_0) == all_image_frame.end())
            {
                RCLCPP_WARN(vins::logger(),
                    "Cannot slide initialization window because oldest header %.9f is missing "
                    "from all_image_frame; scheduling a controlled estimator reset", t_0);
                scheduleControlledRestart();
                return;
            }
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                auto it_0 = all_image_frame.find(t_0);
                auto erase_end = std::next(it_0);
                for (auto frame_it = all_image_frame.begin(); frame_it != erase_end; ++frame_it)
                {
                    delete frame_it->second.pre_integration;
                    frame_it->second.pre_integration = nullptr;
                }
                all_image_frame.erase(all_image_frame.begin(), erase_end);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

bool Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    const double dt = t - latest_time;
    if (!std::isfinite(dt) || !(dt > 0.0) ||
        !linear_acceleration.allFinite() || !angular_velocity.allFinite())
        return false;
    if (latest_schur_active_)
    {
        // An IMU outage is distinct from a camera outage. Do not label an
        // unpropagated/invalid estimate with a new timestamp after a clock jump.
        if (dt > 1.0)
        {
            latest_state_valid_ = false;
            return false;
        }
        latest_schur_state_.propagate(linear_acceleration, angular_velocity, dt,
                                      latest_schur_noise_scale_);
        const auto &imu = latest_schur_state_.imu();
        latest_P = imu.p;
        latest_Q = Eigen::Quaterniond(imu.q[0], imu.q[1], imu.q[2], imu.q[3]);
        latest_V = imu.v;
        latest_Ba = imu.ba;
        latest_Bg = imu.bg;
    }
    else
    {
        Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - latest_gravity_;
        Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
        latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
        Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - latest_gravity_;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        latest_P += dt * latest_V + 0.5 * dt * dt * un_acc;
        latest_V += dt * un_acc;
    }
    latest_time = t;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
    return true;
}

void Estimator::updateLatestStates()
{
    std::lock_guard<std::mutex> propagate_lock(mPropagate);
    latest_state_valid_ = solver_flag == NON_LINEAR && !restart_requested_ &&
                          !suppress_state_publication_;
    latest_schur_active_ = latest_state_valid_ && schur_ekf_active_;
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    latest_gravity_ = g;
    if (latest_schur_active_)
    {
        latest_schur_state_.copyLiveFrom(schur_state_);
        latest_time = latest_schur_state_.timestamp();
        const auto &imu = latest_schur_state_.imu();
        latest_P = imu.p;
        latest_Q = Eigen::Quaterniond(imu.q[0], imu.q[1], imu.q[2], imu.q[3]);
        latest_V = imu.v;
        latest_Ba = imu.ba;
        latest_Bg = imu.bg;
        latest_schur_noise_scale_ = schur_hold_.active() ? 1.0 :
                                  schur_starvation_monitor_.noise_scale();
    }
    if (!latest_state_valid_) return;
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf, tmp_gyrBuf;
    {
        std::lock_guard<std::mutex> buffer_lock(mBuf);
        tmp_accBuf = accBuf;
        tmp_gyrBuf = gyrBuf;
    }
    while (!tmp_accBuf.empty() && latest_state_valid_)
    {
        // Older/equal samples are deliberately skipped, including the endpoint
        // retained by getIMUInterval. Only the tail after the corrected state
        // is replayed; the image-time EKF is never advanced by this predictor.
        fastPredictIMU(tmp_accBuf.front().first, tmp_accBuf.front().second,
                       tmp_gyrBuf.front().second);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
}
