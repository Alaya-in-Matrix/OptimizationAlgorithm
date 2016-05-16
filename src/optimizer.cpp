#include "optimizer.h"
#include <random>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>
using namespace std;
// #define DEBUG_OPTIMIZER
#ifdef DEBUG_OPTIMIZER
mt19937_64 engine(0);
#else
mt19937_64 engine(random_device{}());
#endif

Paras Optimizer::random_init() const noexcept
{
    Paras init(_ranges.size(), 0);
    for(size_t i = 0; i < _ranges.size(); ++i)
    {
        init[i] = uniform_real_distribution<double>(_ranges[i].first, _ranges[i].second)(engine);
    }
    return init;
}
Solution FibOptimizer::optimize() const noexcept
{
    // 1-D function
    // function shoulde be convex function
    if(_ranges.size() != 1)
    {
        return Solution({}, "FibOptimizer requires 1D function, while the actual dim is " + to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if(a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }
    
    const size_t iter = 16;
    vector<double> fib_list{1, 1};
    if(iter > 2)
    {
        fib_list.reserve(iter);
        for(size_t i = 2; i < iter; ++i)
            fib_list.push_back(fib_list[i-1] + fib_list[i-2]);
    }

    for(size_t i = iter - 1; i >0; --i)
    {
        const double rate = fib_list[i - 1] / fib_list[i];

        const double y1 = _func({a1}).fom();
        const double y2 = _func({a2}).fom();

        if(y1 < y2)
            a2 = a1 + rate * (a2 - a1);
        else
            a1 = a2 + rate * (a1 - a2);
    }

    return _func({a1});
}
Solution GoldenSelection::optimize() const noexcept
{
    // 1-D function
    // function shoulde be convex function
    if(_ranges.size() != 1)
    {
        return Solution({}, "GoldenSelection requires 1D function, while the actual dim is " + to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if(a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }
    
    const size_t iter = 16;
    const double rate = (sqrt(5) - 1) / 2;

    for(size_t i = iter - 1; i >0; --i)
    {

        const double y1 = _func({a1}).fom();
        const double y2 = _func({a2}).fom();

        if(y1 < y2)
            a2 = a1 + rate * (a2 - a1);
        else
            a1 = a2 + rate * (a1 - a2);
        // printf("[%g, %g]\n", a1, a2);
    }

    return _func({a1});
}
Solution Extrapolation::optimize() const noexcept
{
    // 1-D function
    // function shoulde be convex function
    if(_ranges.size() != 1)
    {
        return Solution({}, "Extrapolation requires 1D function, while the actual dim is " + to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if(a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }
    double step = 0.01 * (a2 - a1);
    double x1   = _init[0];
    double x2   = x1 + step;
    double y1   = _func({x1}).fom();
    double y2   = _func({x2}).fom();
    if(y2 > y1)
    {
        std::swap(x1, x2);
        std::swap(y1, y2);
        step *= -1;
    }
    double factor = 2;
    double x3     = x2 + factor * step;
    double y3     = _func({x3}).fom();
    while(y3 < y2 && (a1 <= x3 && x3 <= a2))
    {
        factor *= 2;
        x3 += factor * step;
        y3 = _func({x3}).fom();
        double tmp;
    }
    // double xm = x3 - (factor / 2) * step;
    // double ym = _func({xm}).fom();
    double xtmp1 = x3 - factor * step;
    double xtmp2 = x3 - (factor / 2) * step;
    double ytmp1 = _func({xtmp1}).fom();
    double ytmp2 = _func({xtmp2}).fom();
    double xa, xc;
    double ya, yc;
    if(ytmp1 < ytmp2)
    {
        xa = x2;
        xc = xtmp2;
        ya = y2;
        yc = ytmp2;
    }
    else
    {
        xa = xtmp1;
        xc = x3;
        ya = ytmp1;
        yc = y3;
    }

    if(xa > xc)
    {
        std::swap(xa, xc);
        std::swap(ya, yc);
    }
    GoldenSelection gso(_func, {{xa, xc}});
    return gso.optimize();
}
vector<double>GradientMethod::get_gradient(const Paras& p) const noexcept
{
    assert(_epsilon > 0);
    const size_t dim = _ranges.size();
    vector<double> grad(dim, 0);
    const double y = _func(p).fom();
    for(size_t i = 0; i < p.size(); ++i)
    {
        Paras pp = p;
        pp[i] += _epsilon;
        const double yy = _func(pp).fom();
        grad[i] = (yy - y) / _epsilon;
    }
    return grad;
}
bool GradientMethod::in_range(const Paras& p) const noexcept
{
    assert(p.size() == _ranges.size());
    for(size_t i = 0; i < p.size(); ++i)
    {
        const double lb = _ranges[i].first;
        const double ub = _ranges[i].second;
        assert(lb <= ub);

        const double x  = p[i];
        if(! (lb <= x && x <= ub)) return false;
    }
    return true;
}
double GradientMethod::vec_norm(const vector<double>& vec) const noexcept
{
    double sum_square = 0;
    for(auto v : vec)
        sum_square += v * v;
    return sqrt(sum_square);
}
