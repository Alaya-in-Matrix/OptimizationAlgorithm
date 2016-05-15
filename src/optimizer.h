#pragma once
#include "obj.h"
#include <utility>
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
public:
    FibOptimizer(ObjFunc f, Range r) noexcept : Optimizer(f, r) {}
    Solution optimize() const noexcept;
    ~FibOptimizer() {}
};
class GoldenSelection : public Optimizer
{
public:
    GoldenSelection(ObjFunc f, Range r) noexcept : Optimizer(f, r) {}
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
