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

public:
    Optimizer(ObjFunc func, Range r) noexcept : _func(func), _ranges(r), _init{} {}
    Optimizer(ObjFunc func, Range r, Paras i) noexcept : _func(func), _ranges(r), _init(i) {}
    virtual Solution optimize() const noexcept = 0;
    virtual ~Optimizer(){};
};

class FibOptimizer : public Optimizer
{
public:
    FibOptimizer(ObjFunc f, Range r) noexcept : Optimizer(f, r) {}
    Solution optimize() const noexcept;
    ~FibOptimizer(){}
};
