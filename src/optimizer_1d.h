#pragma once
#include "def.h"
#include "obj.h"
#include "Eigen/Dense"
#include "util.h"
#include <utility>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cmath>
class Optimizer1D
{
protected:
    ObjFunc _func;

public:
    Optimizer1D(ObjFunc func) noexcept : _func(func) {}
    virtual Solution optimize() noexcept = 0;
};

class FibOptimizer : public Optimizer1D
{
    const double _lb;
    const double _ub;
    const size_t _iter;

public:
    FibOptimizer(ObjFunc f, double lb, double ub, size_t iter = 16) noexcept;
    Solution optimize() noexcept;
    ~FibOptimizer() {}
};
class GoldenSelection : public Optimizer1D
{
    const double _lb;
    const double _ub;
    const size_t _iter;

public:
    GoldenSelection(ObjFunc f, double lb, double ub, size_t iter = 16) noexcept;
    Solution optimize() noexcept;
    ~GoldenSelection() {}
};
class Extrapolation : public Optimizer1D
{
    const Paras  _init;
    const double _min_len;  // min extrapolation step
    const double _max_len;  // max extrapolation step
public:
    Extrapolation(ObjFunc f, Paras i, double min_len, double max_len) noexcept;
    Solution optimize() noexcept;
    ~Extrapolation() {}
};
