#pragma once
#include "def.h"
#include "obj.h"
#include "Eigen/Dense"
#include "linear_algebra.h"
#include <utility>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cmath>
class Optimizer
{
protected:
    typedef std::vector<std::pair<double, double>> Range;
    ObjFunc _func;
    Range _ranges;
    Paras _init;
    Paras random_init() const noexcept;

public:
    Optimizer(ObjFunc func, Range r) noexcept : _func(func), _ranges(r), _init(random_init()) {}
    Optimizer(ObjFunc func, Range r, Paras i) noexcept : _func(func), _ranges(r), _init(i) {}
    virtual Solution optimize() noexcept = 0;
};

class FibOptimizer : public Optimizer
{
    const size_t _iter;

public:
    FibOptimizer(ObjFunc f, Range r, size_t iter = 16) noexcept : Optimizer(f, r), _iter(iter) {}
    Solution optimize() noexcept;
    ~FibOptimizer() {}
};
class GoldenSelection : public Optimizer
{
    const size_t _iter;

public:
    GoldenSelection(ObjFunc f, Range r, size_t iter = 16) noexcept : Optimizer(f, r), _iter(iter) {}
    Solution optimize() noexcept;
    ~GoldenSelection() {}
};
class Extrapolation : public Optimizer
{
public:
    Extrapolation(ObjFunc f, Range r, Paras i) noexcept : Optimizer(f, r, i) {}
    Solution optimize() noexcept;
    ~Extrapolation() {}
};

class GradientMethod : public Optimizer
{
protected:
    const double        _epsilon; // use _epsilon to calc gradient
    const double        _zero_grad; // threshold to judge whether gradient is zero
    const double        _min_walk; // minimum walk len during iteration
    const size_t        _max_iter;
    const size_t        _dim;
    const std::string   _func_name;
    const std::string   _algo_name;
    std::ofstream _log;
    size_t        _counter; // counter of line search

    virtual std::vector<double> get_gradient(const Paras& p) const noexcept;
    virtual std::vector<double> get_gradient(ObjFunc, const Paras&) const noexcept;
    virtual Eigen::MatrixXd hessian(const Paras& point) const noexcept;
    virtual Solution line_search(const Paras& point, const std::vector<double>& direc) const noexcept;

public:
    void clear_counter() noexcept { _counter = 0; }
    GradientMethod(ObjFunc f, Range r, Paras i, double epsi, double zgrad, double mwalk, size_t max_iter,
                      std::string fname, std::string aname) noexcept;
    size_t counter() const noexcept { return _counter; } 
    ~GradientMethod() { if(_log.is_open()) _log.close(); } 
};

#define TYPICAL_DEF(ClassName)                                                      \
    ClassName(ObjFunc f, Range r, Paras i, double epsi, double zgrad, double mwalk, \
              size_t max_iter, std::string fname, std::string aname)                \
        : GradientMethod(f, r, i, epsi, zgrad, mwalk, max_iter, fname, aname)       \
    {                                                                               \
    }                                                                               \
    Solution optimize() noexcept;                                                   \
    ~ClassName() {}
class GradientDescent : public GradientMethod
{
    void write_log(Paras& p, double fom, std::vector<double>& grad) noexcept;
public:
    TYPICAL_DEF(GradientDescent);
};
class ConjugateGradient : public GradientMethod
{
    void write_log(Paras& p, double fom, std::vector<double>& grad, std::vector<double>& conj_grad) noexcept;
public:
    TYPICAL_DEF(ConjugateGradient);
};
class Newton : public GradientMethod
{
    void write_log(Paras& p, double fom, std::vector<double>& grad, Eigen::MatrixXd& hess) noexcept;
public:
    TYPICAL_DEF(Newton);
};
class DFP : public GradientMethod
{
    void write_log(Paras& p, double fom, std::vector<double>& grad, Eigen::MatrixXd& quasi_hess) noexcept;
public:
    TYPICAL_DEF(DFP);
};
class BFGS : public GradientMethod
{
    void write_log(Paras& p, double fom, std::vector<double>& grad, Eigen::MatrixXd& quasi_hess) noexcept;
public:
    TYPICAL_DEF(BFGS);
};
