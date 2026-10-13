#include "camodocal/camera_models/CameraFactory.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

#include <yaml-cpp/yaml.h>

#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/PinholeFullCamera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include "ceres/ceres.h"

namespace camodocal
{

boost::shared_ptr< CameraFactory > CameraFactory::m_instance;

namespace
{

// Reads a whole file into a string; returns an empty string on failure.
std::string
readFileText( const std::string& filename )
{
    std::ifstream stream( filename, std::ios::binary );
    if ( !stream.is_open( ) )
    {
        return std::string( );
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf( );
    return buffer.str( );
}

// A ROS sensor_msgs/CameraInfo dump has no OpenCV "%YAML" header and no
// camodocal "model_type", but always carries a "camera_matrix".
bool
looksLikeCameraInfo( const std::string& text )
{
    return text.find( "camera_matrix" ) != std::string::npos
           && text.find( "model_type" ) == std::string::npos;
}

std::vector< double >
cameraInfoData( const YAML::Node& node )
{
    if ( !node || !node.IsMap( ) || !node["data"] )
    {
        return std::vector< double >( );
    }
    try
    {
        return node["data"].as< std::vector< double > >( );
    }
    catch ( const YAML::Exception& )
    {
        return std::vector< double >( );
    }
}

// Builds a camodocal camera from a ROS sensor_msgs/CameraInfo YAML file.
// plumb_bob / rational_polynomial map to the pinhole model with
// radial-tangential distortion; equidistant / fisheye / kannala_brandt map to
// the Kannala-Brandt (equidistant) model.
CameraPtr
cameraFromCameraInfo( const std::string& filename )
{
    YAML::Node root;
    try
    {
        root = YAML::LoadFile( filename );
    }
    catch ( const YAML::Exception& e )
    {
        std::cerr << "# ERROR: Failed to parse CameraInfo file " << filename << ": "
                  << e.what( ) << std::endl;
        return CameraPtr( );
    }

    const std::vector< double > K = cameraInfoData( root["camera_matrix"] );
    if ( K.size( ) != 9 )
    {
        std::cerr << "# ERROR: CameraInfo file " << filename
                  << " has an invalid camera_matrix (expected 9 values)" << std::endl;
        return CameraPtr( );
    }

    std::string cameraName = "camera";
    if ( root["camera_name"] )
    {
        cameraName = root["camera_name"].as< std::string >( );
    }

    int width  = 0;
    int height = 0;
    if ( root["image_width"] )
    {
        width = root["image_width"].as< int >( );
    }
    if ( root["image_height"] )
    {
        height = root["image_height"].as< int >( );
    }

    std::string model = "plumb_bob";
    if ( root["distortion_model"] )
    {
        model = root["distortion_model"].as< std::string >( );
    }

    std::vector< double > distortion = cameraInfoData( root["distortion_coefficients"] );
    distortion.resize( 5, 0.0 );

    const double fx = K[0];
    const double fy = K[4];
    const double cx = K[2];
    const double cy = K[5];

    std::string lower = model;
    std::transform( lower.begin( ), lower.end( ), lower.begin( ),
                    []( unsigned char c ) { return static_cast< char >( std::tolower( c ) ); } );

    if ( lower == "equidistant" || lower == "fisheye" || lower == "kannala_brandt" )
    {
        EquidistantCameraPtr camera( new EquidistantCamera );
        EquidistantCamera::Parameters params = camera->getParameters( );
        params.cameraName( )  = cameraName;
        params.imageWidth( )  = width;
        params.imageHeight( ) = height;
        params.k2( )          = distortion[0];
        params.k3( )          = distortion[1];
        params.k4( )          = distortion[2];
        params.k5( )          = distortion[3];
        params.mu( )          = fx;
        params.mv( )          = fy;
        params.u0( )          = cx;
        params.v0( )          = cy;
        camera->setParameters( params );
        return camera;
    }

    PinholeCameraPtr camera( new PinholeCamera );
    PinholeCamera::Parameters params = camera->getParameters( );
    params.cameraName( )  = cameraName;
    params.imageWidth( )  = width;
    params.imageHeight( ) = height;
    params.k1( )          = distortion[0];
    params.k2( )          = distortion[1];
    params.p1( )          = distortion[2];
    params.p2( )          = distortion[3];
    params.fx( )          = fx;
    params.fy( )          = fy;
    params.cx( )          = cx;
    params.cy( )          = cy;
    camera->setParameters( params );
    return camera;
}


}  // namespace

CameraFactory::CameraFactory( ) {}

boost::shared_ptr< CameraFactory >
CameraFactory::instance( void )
{
    if ( m_instance.get( ) == 0 )
    {
        m_instance.reset( new CameraFactory );
    }

    return m_instance;
}

CameraPtr
CameraFactory::generateCamera( Camera::ModelType modelType, const std::string& cameraName, cv::Size imageSize ) const
{
    switch ( modelType )
    {
        case Camera::KANNALA_BRANDT:
        {
            EquidistantCameraPtr camera( new EquidistantCamera );

            EquidistantCamera::Parameters params = camera->getParameters( );
            params.cameraName( )                 = cameraName;
            params.imageWidth( )                 = imageSize.width;
            params.imageHeight( )                = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE:
        {
            PinholeCameraPtr camera( new PinholeCamera );

            PinholeCamera::Parameters params = camera->getParameters( );
            params.cameraName( )             = cameraName;
            params.imageWidth( )             = imageSize.width;
            params.imageHeight( )            = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE_FULL:
        {
            PinholeFullCameraPtr camera( new PinholeFullCamera );

            PinholeFullCamera::Parameters params = camera->getParameters( );
            params.cameraName( )                 = cameraName;
            params.imageWidth( )                 = imageSize.width;
            params.imageHeight( )                = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::SCARAMUZZA:
        {
            OCAMCameraPtr camera( new OCAMCamera );

            OCAMCamera::Parameters params = camera->getParameters( );
            params.cameraName( )          = cameraName;
            params.imageWidth( )          = imageSize.width;
            params.imageHeight( )         = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::MEI:
        default:
        {
            CataCameraPtr camera( new CataCamera );

            CataCamera::Parameters params = camera->getParameters( );
            params.cameraName( )          = cameraName;
            params.imageWidth( )          = imageSize.width;
            params.imageHeight( )         = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
    }
}

CameraPtr
CameraFactory::generateCameraFromYamlFile( const std::string& filename )
{
    const std::string text = readFileText( filename );
    if ( text.empty( ) )
    {
        return CameraPtr( );
    }

    // ROS sensor_msgs/CameraInfo files cannot be parsed by cv::FileStorage, so
    // detect and convert them with yaml-cpp before falling back to OpenCV.
    if ( looksLikeCameraInfo( text ) )
    {
        return cameraFromCameraInfo( filename );
    }

    cv::FileStorage fs( filename, cv::FileStorage::READ );

    if ( !fs.isOpened( ) )
    {
        return CameraPtr( );
    }

    Camera::ModelType modelType = Camera::MEI;
    if ( !fs["model_type"].isNone( ) )
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if ( boost::iequals( sModelType, "kannala_brandt" ) )
        {
            modelType = Camera::KANNALA_BRANDT;
        }
        else if ( boost::iequals( sModelType, "mei" ) )
        {
            modelType = Camera::MEI;
        }
        else if ( boost::iequals( sModelType, "scaramuzza" ) )
        {
            modelType = Camera::SCARAMUZZA;
        }
        else if ( boost::iequals( sModelType, "pinhole" ) )
        {
            modelType = Camera::PINHOLE;
        }
        else if ( boost::iequals( sModelType, "PINHOLE_FULL" ) )
        {
            modelType = Camera::PINHOLE_FULL;
        }
        else
        {
            std::cerr << "# ERROR: Unknown camera model: " << sModelType << std::endl;
            return CameraPtr( );
        }
    }

    switch ( modelType )
    {
        case Camera::KANNALA_BRANDT:
        {
            EquidistantCameraPtr camera( new EquidistantCamera );

            EquidistantCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE:
        {
            PinholeCameraPtr camera( new PinholeCamera );

            PinholeCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE_FULL:
        {
            PinholeFullCameraPtr camera( new PinholeFullCamera );

            PinholeFullCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::SCARAMUZZA:
        {
            OCAMCameraPtr camera( new OCAMCamera );

            OCAMCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::MEI:
        default:
        {
            CataCameraPtr camera( new CataCamera );

            CataCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
    }

    return CameraPtr( );
}
}
