#pragma once
#include "obj.h"
#include <utility>
#include <iostream>
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
    virtual Solution optimize() const noexcept = 0;
    virtual ~Optimizer(){};
};

class FibOptimizer : public Optimizer
{
    const size_t _iter;
public:
    FibOptimizer(ObjFunc f, Range r, size_t iter = 16) noexcept : Optimizer(f, r), _iter(iter) {}
    Solution optimize() const noexcept;
    ~FibOptimizer() {}
};
class GoldenSelection : public Optimizer
{
    const size_t _iter;
public:
    GoldenSelection(ObjFunc f, Range r, size_t iter = 16) noexcept : Optimizer(f, r), _iter(iter) {}
    Solution optimize() const noexcept;
    ~GoldenSelection() {}
};
class Extrapolation : public Optimizer
{
public:
    Extrapolation(ObjFunc f, Range r) noexcept : Optimizer(f, r) {}
    Solution optimize() const noexcept;
    ~Extrapolation() {}
};

// Abstract class for all gradient method: GradientDescent\Newton\BFGS...
class GradientMethod : public Optimizer
{
protected:
    const double _epsilon;
    double vec_norm(const std::vector<double>& vec) const noexcept;
    bool in_range(const Paras& p) const noexcept;
    virtual std::vector<double> get_gradient(const Paras& p) const noexcept;

public:
    GradientMethod(ObjFunc f, Range r, double epsilon) noexcept : Optimizer(f, r), _epsilon(epsilon)
    {
        if (_epsilon <= 0)
        {
            std::cerr << "epsilon <= 0" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    virtual ~GradientMethod() {}
};
class GradientDescent : public GradientMethod
{
public:
    GradientDescent(ObjFunc f, Range r, double epsilon) noexcept : GradientMethod(f, r, epsilon) {}
    Solution optimize() const noexcept;
    ~GradientDescent() {}
};
