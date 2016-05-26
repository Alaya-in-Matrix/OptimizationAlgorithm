#pragma once
#include "def.h"
#include "obj.h"
#include "Eigen/Dense"
#include "linear_algebra.h"
#include <utility>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cmath>
class Optimizer
{
protected:
    ObjFunc _func;
    size_t  _dim;

public:
    Optimizer(ObjFunc func, size_t d) noexcept : _func(func), _dim(d) {}
    virtual Solution optimize() noexcept = 0;
};

class FibOptimizer : public Optimizer
{
    const double _lb;
    const double _ub;
    const size_t _iter;

public:
    FibOptimizer(ObjFunc f, double lb, double ub, size_t iter = 16) noexcept : Optimizer(f, 1),
                                                                               _lb(lb),
                                                                               _ub(ub),
                                                                               _iter(iter)
    {}
    Solution optimize() noexcept;
    ~FibOptimizer() {}
};
class GoldenSelection : public Optimizer
{
    const double _lb;
    const double _ub;
    const size_t _iter;

public:
    GoldenSelection(ObjFunc f, double lb, double ub, size_t iter = 16) noexcept
        : Optimizer(f, 1),
          _lb(lb),
          _ub(ub),
          _iter(iter)
    {}
    Solution optimize() noexcept;
    ~GoldenSelection() {}
};
class Extrapolation : public Optimizer
{
    const Paras  _init;
    const double _min_len;  // min extrapolation step
    const double _max_len;  // max extrapolation step
public:
    Extrapolation(ObjFunc f, Paras i, double min_len, double max_len) noexcept : Optimizer(f, 1),
                                                                                 _init(i),
                                                                                 _min_len(min_len),
                                                                                 _max_len(max_len)
    {
        if(!(min_len > 0 && max_len > 0 && min_len < max_len))
        {
            std::cerr << "Not satisfied: min_len > 0 && max_len > 0 && min_len < max_len"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    Solution optimize() noexcept;
    ~Extrapolation() {}
};
Solution line_search(ObjFunc func, const Paras& point, const Eigen::VectorXd& direc,
                     double min_walk, double max_walk) noexcept;
class GradientMethod : public Optimizer
{
protected:
    const Paras         _init;
    const double        _epsilon; // use _epsilon to calc gradient
    const double        _zero_grad; // threshold to judge whether gradient is zero
    const double        _min_walk; // minimum walk len during iteration
    const double        _max_walk; // minimum walk len during iteration
    const size_t        _max_iter;
    const std::string   _func_name;
    const std::string   _algo_name;

    std::ofstream _log;
    size_t _counter;  // counter of line search

    virtual Eigen::VectorXd get_gradient(const Paras& p) const noexcept;
    virtual Eigen::VectorXd get_gradient(ObjFunc, const Paras&) const noexcept;
    virtual Eigen::MatrixXd hessian(const Paras& point) const noexcept;

public:
    void clear_counter() noexcept { _counter = 0; }
    GradientMethod(ObjFunc f, size_t d, Paras i, double epsi, double zgrad, double minwalk,
                   double maxwalk, size_t max_iter, std::string fname, std::string aname) noexcept;
    size_t counter() const noexcept { return _counter; } 
    ~GradientMethod() { if(_log.is_open()) _log.close(); } 
};

#define TYPICAL_DEF(ClassName)                                                               \
    ClassName(ObjFunc f, size_t d, Paras i, double epsi, double zgrad, double minwalk,       \
              double maxwalk, size_t max_iter, std::string fname)                            \
        : GradientMethod(f, d, i, epsi, zgrad, minwalk, maxwalk, max_iter, fname, #ClassName) \
    {                                                                                         \
    }                                                                                         \
    Solution optimize() noexcept;                                                             \
    ~ClassName() {}
class GradientDescent : public GradientMethod
{
    void write_log(Paras& p, double fom, Eigen::VectorXd& grad) noexcept;
public:
    TYPICAL_DEF(GradientDescent);
};
class ConjugateGradient : public GradientMethod
{
    void write_log(Paras& p, double fom, Eigen::VectorXd& grad, Eigen::VectorXd& conj_grad) noexcept;
public:
    TYPICAL_DEF(ConjugateGradient);
};
class Newton : public GradientMethod
{
    void write_log(Paras& p, double fom, Eigen::VectorXd& grad, Eigen::MatrixXd& hess) noexcept;
public:
    TYPICAL_DEF(Newton);
};
class DFP : public GradientMethod
{
    void write_log(Paras& p, double fom, Eigen::VectorXd& grad, Eigen::MatrixXd& quasi_hess) noexcept;
public:
    TYPICAL_DEF(DFP);
};
class BFGS : public GradientMethod
{
    void write_log(Paras& p, double fom, Eigen::VectorXd& grad, Eigen::MatrixXd& quasi_hess) noexcept;
public:
    TYPICAL_DEF(BFGS);
};
class NelderMead : public Optimizer
{
    const double _alpha;
    const double _gamma;
    const double _rho;
    const double _sigma;
    const double _converge_len;
    const size_t _max_iter;
    const std::string _func_name;

    std::ofstream         _log;
    std::vector<Solution> _points;
    size_t                _counter;

    double max_simplex_len() const noexcept;
    void write_log(const Solution& s) noexcept;

public:
    size_t counter() const noexcept { return _counter; } 
    NelderMead(ObjFunc f, size_t d, std::vector<Paras> i, double a, double g, double r, double s,
               double conv_len, size_t max_iter, std::string fname) noexcept;
    Solution optimize() noexcept;
};
class Powell : public Optimizer
{
    Paras _init;

    const size_t _max_iter;
    const double _min_walk;
    const double _max_walk;
    const std::string _func_name;

    size_t _counter;
    std::ofstream _log;
    void write_log(const Solution& s) noexcept;

public:
    size_t counter() const noexcept { return _counter; }
    Powell(ObjFunc f, size_t d, const Paras& i, size_t max_iter, double min_walk, double max_walk,
           std::string fname) noexcept;
    Solution optimize() noexcept;
};
