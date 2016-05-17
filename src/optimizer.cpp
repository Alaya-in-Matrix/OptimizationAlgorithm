#include "def.h"
#include "optimizer.h"
#include "linear_algebra.h"
#include <random>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>
using namespace std;
mt19937_64 engine(RAND_SEED);

Paras Optimizer::random_init() const noexcept
{
    Paras init(_ranges.size(), 0);
    for (size_t i = 0; i < _ranges.size(); ++i)
    {
        init[i] = uniform_real_distribution<double>(_ranges[i].first, _ranges[i].second)(engine);
    }
    return init;
}
Solution FibOptimizer::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    if (_ranges.size() != 1)
    {
        return Solution({}, "FibOptimizer requires 1D function, while the actual dim is " +
                                to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if (a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }

    vector<double> fib_list{1, 1};
    if (_iter > 2)
    {
        fib_list.reserve(_iter);
        for (size_t i = 2; i < _iter; ++i) fib_list.push_back(fib_list[i - 1] + fib_list[i - 2]);
    }

    for (size_t i = _iter - 1; i > 0; --i)
    {
        const double rate = fib_list[i - 1] / fib_list[i];

        const double y1 = _func({a1}).fom();
        const double y2 = _func({a2}).fom();

        if (y1 < y2)
            a2 = a1 + rate * (a2 - a1);
        else
            a1 = a2 + rate * (a1 - a2);
    }
    return _func({a1});
}
Solution GoldenSelection::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    if (_ranges.size() != 1)
    {
        return Solution({}, "GoldenSelection requires 1D function, while the actual dim is " +
                                to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if (a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }

    const double rate = (sqrt(5) - 1) / 2;

    double y1, y2;
    for (size_t i = _iter - 1; i > 0; --i)
    {
        y1 = _func({a1}).fom();
        y2 = _func({a2}).fom();

        if (y1 < y2)
            a2 = a1 + rate * (a2 - a1);
        else
            a1 = a2 + rate * (a1 - a2);
    }
    return _func({a1});
}
Solution Extrapolation::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    if (_ranges.size() != 1)
    {
        return Solution({}, "Extrapolation requires 1D function, while the actual dim is " +
                                to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if (a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }
    double step = 0.01 * (a2 - a1);
    double x1 = _init[0];
    double x2 = x1 + step;
    double y1 = _func({x1}).fom();
    double y2 = _func({x2}).fom();
    if (y2 > y1)
    {
        std::swap(x1, x2);
        std::swap(y1, y2);
        step *= -1;
    }
    double factor = 2;
    double x3 = x2 + factor * step;
    double y3 = _func({x3}).fom();
    while (y3 < y2 && (a1 <= x3 && x3 <= a2))
    {
        factor *= 2;
        x3 += factor * step;
        y3 = _func({x3}).fom();
    }
    double xtmp1 = x3 - factor * step;
    double xtmp2 = x3 - (factor / 2) * step;
    double ytmp1 = _func({xtmp1}).fom();
    double ytmp2 = _func({xtmp2}).fom();
    double xa, xc;
    double ya, yc;
    if (ytmp1 < ytmp2)
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

    if (xa > xc)
    {
        std::swap(xa, xc);
        std::swap(ya, yc);
    }
    GoldenSelection gso(_func, {{xa, xc}});
    return gso.optimize();
}
vector<double> MultiDimOptimizer::get_gradient(const Paras& p) const noexcept
{
    const size_t dim = _ranges.size();

    assert(_epsilon > 0);
    assert(p.size() == dim);
    vector<double> grad(dim, 0);
    const double y = _func(p).fom();
    for (size_t i = 0; i < p.size(); ++i)
    {
        Paras pp = p;
        pp[i] += _epsilon;
        const double yy = _func(pp).fom();
        grad[i] = (yy - y) / _epsilon;
    }
    return grad;
}
bool MultiDimOptimizer::in_range(const Paras& p) const noexcept
{
    assert(p.size() == _ranges.size());
    for (size_t i = 0; i < p.size(); ++i)
    {
        const double lb = _ranges[i].first;
        const double ub = _ranges[i].second;
        assert(lb <= ub);

        const double x = p[i];
        if(x - lb < _epsilon || ub - x < _epsilon)
            return false;
    }
    return true;
}
Solution MultiDimOptimizer::line_search(const Paras& point, const vector<double>& direction) const noexcept
{
    double max_step  = 1;
    const double dim = _ranges.size();
    assert(point.size() == dim && direction.size() == dim);

    for(size_t i = 0; i < dim; ++i)
    {
        const double lb = _ranges[i].first;
        const double ub = _ranges[i].second;
        const double g  = direction[i];
        const double x  = point[i];
        double step_ub;
        if(fabs(g) < _epsilon)
            step_ub = numeric_limits<double>::infinity();
        else 
            step_ub = g > 0 ? (ub - x) / g : (lb - x) / g;

        if(max_step > step_ub)
            max_step = step_ub;
    }
    size_t gs_iter;
    double rate = _epsilon / (max_step * vec_norm(direction));
    if(rate > 0.618)
        gs_iter = 2;
    else
    {
        gs_iter = 1 + log10(rate) / log10(0.618);
    }
    GoldenSelection gso(
        [&](const vector<double> step) -> Solution
        {
            auto debug = point + step[0] * direction;
            Solution y = _func(point + step[0] * direction);
            return _func(point + step[0] * direction);
        },
        {{0, max_step}}, gs_iter);
    return gso.optimize();
}
Solution GradientDescent::optimize() noexcept
{
    _counter = 0;
    Paras point = _init;
    const double zero_grad = 1e-2;
    vector<double> grad    = get_gradient(point);
    double grad_norm       = vec_norm(grad);
    while (grad_norm > zero_grad && in_range(point))
    {
        const vector<double> direction = -1 * grad;

        point        = line_search(point, direction).solution();
        grad         = get_gradient(point);
        grad_norm    = vec_norm(grad);
        ++_counter;
    }
    if(! in_range(point))
        cerr << "out of range" << endl;
    return _func(point);
}
Solution ConjugateGradient::optimize() noexcept
{
    _counter = 0;
    const size_t dim         = _ranges.size();
    const double zero_grad   = 1e-2;
    Paras point              = _init;
    vector<double> grad      = get_gradient(point);
    double grad_norm         = vec_norm(grad);

    while(grad_norm > zero_grad && in_range(point))
    {
        vector<double> conj_grad = grad;
        for(size_t i = 0; i < dim; ++i)
        {
            const vector<double> direction = -1 * conj_grad;
            const Solution sol             = line_search(point, direction);
            const Paras new_point          = sol.solution();
            const vector<double> new_grad  = get_gradient(new_point);

            conj_grad = new_grad + pow(vec_norm(new_grad) / vec_norm(grad), 2) * conj_grad;
            grad      = new_grad;
            point     = new_point;
            grad_norm = vec_norm(grad);
            ++_counter;

            if(! (grad_norm > zero_grad && in_range(point))) break;
        }
    }
    if(! in_range(point))
        cerr << "out of range" << endl;
    return _func(point);
}
