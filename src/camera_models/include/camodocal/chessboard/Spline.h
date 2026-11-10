/*  dynamo:- Event driven molecular dynamics simulator
    http://www.marcusbannerman.co.uk/dynamo
    Copyright (C) 2011  Marcus N Campbell Bannerman <m.bannerman@gmail.com>

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    version 3 as published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <exception>

namespace ublas = boost::numeric::ublas;

class Spline : private std::vector<std::pair<double, double>> {
public:
  //The boundary conditions available
  // Keep the existing enum representation for interface compatibility.
  // NOLINTNEXTLINE(performance-enum-size)
  enum BcType_e : unsigned int { FIXED_1ST_DERIV_BC, FIXED_2ND_DERIV_BC, PARABOLIC_RUNOUT_BC };

  // Keep the existing enum representation for interface compatibility.
  // NOLINTNEXTLINE(performance-enum-size)
  enum SplineType_e : unsigned int { LINEAR, CUBIC };

  //Constructor takes the boundary conditions as arguments, this
  //sets the first derivative (gradient) at the lower and upper
  //end points
  Spline()
    : valid_(false), BCLow_(FIXED_2ND_DERIV_BC), BCHigh_(FIXED_2ND_DERIV_BC), BCLowVal_(0), BCHighVal_(0),
      type_(CUBIC)
  {
  }

  typedef std::vector<std::pair<double, double>> base;
  typedef base::const_iterator const_iterator;

  //Standard STL read-only container stuff
  const_iterator begin() const { return base::begin(); }
  const_iterator end() const { return base::end(); }
  void clear()
  {
    valid_ = false;
    base::clear();
    data_.clear();
  }
  size_t size() const { return base::size(); }
  size_t max_size() const { return base::max_size(); }
  size_t capacity() const { return base::capacity(); }
  bool empty() const { return base::empty(); }

  //Add a point to the spline, and invalidate it so its
  //recalculated on the next access
  inline void addPoint(double _x, double _y)
  {
    valid_ = false;
    base::push_back(std::pair<double, double>(_x, _y));
  }

  //Reset the boundary conditions
  inline void setLowBC(BcType_e _BC, double _val = 0)
  {
    BCLow_ = _BC;
    BCLowVal_ = _val;
    valid_ = false;
  }

  inline void setHighBC(BcType_e _BC, double _val = 0)
  {
    BCHigh_ = _BC;
    BCHighVal_ = _val;
    valid_ = false;
  }

  void setType(SplineType_e _type)
  {
    _type = _type;
    valid_ = false;
  }

  //Check if the spline has been calculated, then generate the
  //spline interpolated value
  double operator()(double _xval)
  {
    if (!valid_)
      generate();

    //Special cases when we're outside the range of the spline points
    if (_xval <= x(0))
      return lowCalc(_xval);
    if (_xval >= x(size() - 1))
      return highCalc(_xval);

    //Check all intervals except the last one
    for (std::vector<SplineData_s>::const_iterator iPtr = data_.begin(); iPtr != data_.end() - 1; ++iPtr)
      if ((_xval >= iPtr->x_) && (_xval <= (iPtr + 1)->x_))
        return splineCalc(iPtr, _xval);

    return splineCalc(data_.end() - 1, _xval);
  }

private:
  ///////PRIVATE DATA MEMBERS
  struct SplineData_s {
    double x_, a_, b_, c_, d_;
  };
  //vector of calculated spline data
  std::vector<SplineData_s> data_;
  //Second derivative at each point
  ublas::vector<double> ddy_;
  //Tracks whether the spline parameters have been calculated for
  //the current set of points
  bool valid_;
  //The boundary conditions
  BcType_e BCLow_, BCHigh_;
  //The values of the boundary conditions
  double BCLowVal_, BCHighVal_;

  SplineType_e type_;

  ///////PRIVATE FUNCTIONS
  //Function to calculate the value of a given spline at a point xval
  inline double splineCalc(std::vector<SplineData_s>::const_iterator _i, double _xval)
  {
    const double lx = _xval - _i->x_;
    return ((_i->a_ * lx + _i->b_) * lx + _i->c_) * lx + _i->d_;
  }

  inline double lowCalc(double _xval)
  {
    const double lx = _xval - x(0);

    if (type_ == LINEAR)
      return lx * BCHighVal_ + y(0);

    const double firstDeriv = (y(1) - y(0)) / h(0) - 2 * h(0) * (data_[0].b_ + 2 * data_[1].b_) / 6;

    switch (BCLow_) {
    case FIXED_1ST_DERIV_BC:
      return lx * BCLowVal_ + y(0);
    case FIXED_2ND_DERIV_BC:
      return lx * lx * BCLowVal_ + firstDeriv * lx + y(0);
    case PARABOLIC_RUNOUT_BC:
      return lx * lx * ddy_[0] + lx * firstDeriv + y(0);
    }
    throw std::runtime_error("Unknown BC");
  }

  inline double highCalc(double _xval)
  {
    const double lx = _xval - x(size() - 1);

    if (type_ == LINEAR)
      return lx * BCHighVal_ + y(size() - 1);

    const double firstDeriv = 2 * h(size() - 2) * (ddy_[size() - 2] + 2 * ddy_[size() - 1]) / 6 +
                              (y(size() - 1) - y(size() - 2)) / h(size() - 2);

    switch (BCHigh_) {
    case FIXED_1ST_DERIV_BC:
      return lx * BCHighVal_ + y(size() - 1);
    case FIXED_2ND_DERIV_BC:
      return lx * lx * BCHighVal_ + firstDeriv * lx + y(size() - 1);
    case PARABOLIC_RUNOUT_BC:
      return lx * lx * ddy_[size() - 1] + lx * firstDeriv + y(size() - 1);
    }
    throw std::runtime_error("Unknown BC");
  }

  //These just provide access to the point data in a clean way
  inline double x(size_t _i) const { return operator[](_i).first; }
  inline double y(size_t _i) const { return operator[](_i).second; }
  inline double h(size_t _i) const { return x(_i + 1) - x(_i); }

  //Invert a arbitrary matrix using the boost ublas library
  template <class T> bool invertMatrix(ublas::matrix<T> _A, ublas::matrix<T> & _inverse)
  {
    using namespace ublas;

    // create a permutation matrix for the LU-factorization
    permutation_matrix<std::size_t> pm(_A.size1());

    // perform LU-factorization
    int res = lu_factorize(_A, pm);
    if (res != 0)
      return false;

    // create identity matrix of "inverse"
    _inverse.assign(ublas::identity_matrix<T>(_A.size1()));

    // backsubstitute to get the inverse
    lu_substitute(_A, pm, _inverse);

    return true;
  }

  //This function will recalculate the spline parameters and store
  //them in _data, ready for spline interpolation
  void generate()
  {
    if (size() < 2)
      throw std::runtime_error("Spline requires at least 2 points");

    //If any spline points are at the same x location, we have to
    //just slightly seperate them
    {
      bool testPassed(false);
      while (!testPassed) {
        testPassed = true;
        std::sort(base::begin(), base::end());

        for (base::iterator iPtr = base::begin(); iPtr != base::end() - 1; ++iPtr)
          if (iPtr->first == (iPtr + 1)->first) {
            if ((iPtr + 1)->first != 0)
              (iPtr + 1)->first += (iPtr + 1)->first * std::numeric_limits<double>::epsilon() * 10;
            else
              (iPtr + 1)->first = std::numeric_limits<double>::epsilon() * 10;
            testPassed = false;
            break;
          }
      }
    }

    const size_t e = size() - 1;

    switch (type_) {
    case LINEAR: {
      data_.resize(e);
      for (size_t i(0); i < e; ++i) {
        data_[i].x_ = x(i);
        data_[i].a_ = 0;
        data_[i].b_ = 0;
        data_[i].c_ = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
        data_[i].d_ = y(i);
      }
      break;
    }
    case CUBIC: {
      ublas::matrix<double> A(size(), size());
      for (size_t yv(0); yv <= e; ++yv)
        for (size_t xv(0); xv <= e; ++xv)
          A(xv, yv) = 0;

      for (size_t i(1); i < e; ++i) {
        A(i - 1, i) = h(i - 1);
        A(i, i) = 2 * (h(i - 1) + h(i));
        A(i + 1, i) = h(i);
      }

      ublas::vector<double> C(size());
      for (size_t xv(0); xv <= e; ++xv)
        C(xv) = 0;

      for (size_t i(1); i < e; ++i)
        C(i) = 6 * ((y(i + 1) - y(i)) / h(i) - (y(i) - y(i - 1)) / h(i - 1));

      //Boundary conditions
      switch (BCLow_) {
      case FIXED_1ST_DERIV_BC:
        C(0) = 6 * ((y(1) - y(0)) / h(0) - BCLowVal_);
        A(0, 0) = 2 * h(0);
        A(1, 0) = h(0);
        break;
      case FIXED_2ND_DERIV_BC:
        C(0) = BCLowVal_;
        A(0, 0) = 1;
        break;
      case PARABOLIC_RUNOUT_BC:
        C(0) = 0;
        A(0, 0) = 1;
        A(1, 0) = -1;
        break;
      }

      switch (BCHigh_) {
      case FIXED_1ST_DERIV_BC:
        C(e) = 6 * (BCHighVal_ - (y(e) - y(e - 1)) / h(e - 1));
        A(e, e) = 2 * h(e - 1);
        A(e - 1, e) = h(e - 1);
        break;
      case FIXED_2ND_DERIV_BC:
        C(e) = BCHighVal_;
        A(e, e) = 1;
        break;
      case PARABOLIC_RUNOUT_BC:
        C(e) = 0;
        A(e, e) = 1;
        A(e - 1, e) = -1;
        break;
      }

      ublas::matrix<double> AInv(size(), size());
      invertMatrix(A, AInv);

      ddy_ = ublas::prod(C, AInv);

      data_.resize(size() - 1);
      for (size_t i(0); i < e; ++i) {
        data_[i].x_ = x(i);
        data_[i].a_ = (ddy_(i + 1) - ddy_(i)) / (6 * h(i));
        data_[i].b_ = ddy_(i) / 2;
        data_[i].c_ = (y(i + 1) - y(i)) / h(i) - ddy_(i + 1) * h(i) / 6 - ddy_(i) * h(i) / 3;
        data_[i].d_ = y(i);
      }
    }
    }
    valid_ = true;
  }
};
