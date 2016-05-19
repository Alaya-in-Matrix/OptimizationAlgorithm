#pragma once
#include "obj.h"
#include <utility>
#include <iostream>
#include <fstream>
#include "Eigen/Dense"
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
    Extrapolation(ObjFunc f, Range r) noexcept : Optimizer(f, r) {}
    Solution optimize() noexcept;
    ~Extrapolation() {}
};

class MultiDimOptimizer : public Optimizer
{
protected:
    const double _epsilon; // use _epsilon to calc gradient
    size_t _counter; // counter of line search
    bool in_range(const Paras& p) const noexcept;
    std::vector<double> get_gradient(const Paras& p) const noexcept;
    std::vector<double> get_gradient(ObjFunc, const Paras&) const noexcept;
    Solution line_search(const Paras& point, const std::vector<double>& direc) const noexcept;
    std::ofstream _log;

public:
    MultiDimOptimizer(ObjFunc f, Range r, double epsilon) noexcept;
    MultiDimOptimizer(ObjFunc f, Range r, Paras i, double epsilon) noexcept;
    size_t counter() const noexcept { return _counter; } 
    virtual ~MultiDimOptimizer() {
        _log.close();
    }
};
#define TYPICAL_DEF(ClassName)                                                                                  \
    ClassName(ObjFunc f, Range r, double epsilon) noexcept : MultiDimOptimizer(f, r, epsilon) {}                \
    ClassName(ObjFunc f, Range r, Paras i, double epsilon) noexcept : MultiDimOptimizer(f, r, i, epsilon) {}    \
    Solution optimize() noexcept;                                                                               \
    ~ClassName() {} 
class GradientDescent : public MultiDimOptimizer
{
public:
    TYPICAL_DEF(GradientDescent);
};
class ConjugateGradient : public MultiDimOptimizer
{
public:
    TYPICAL_DEF(ConjugateGradient);
};
class Newton : public MultiDimOptimizer
{
    Eigen::MatrixXd hessian(const Paras& point) const noexcept;

public:
    TYPICAL_DEF(Newton);
};
