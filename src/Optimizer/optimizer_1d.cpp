#include "optimizer_1d.h"
#include <algorithm>
using namespace std;
using namespace Eigen;
FibOptimizer::FibOptimizer(ObjFunc f, double lb, double ub, size_t iter) noexcept
    : Optimizer1D(f),
      _lb(lb),
      _ub(ub),
      _iter(iter)
{}
Solution FibOptimizer::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    double a1 = _lb;
    double a2 = _ub;
    if (a1 > a2)
    {
        cerr << ("Range is [" + to_string(a1) + ", " + to_string(a2) + "]") << endl;
        exit(EXIT_FAILURE);
    }

    vector<double> fib_list{1, 1};
    if (_iter > 2)
        for (size_t i = 2; i < _iter + 1; ++i) fib_list.push_back(fib_list[i - 1] + fib_list[i - 2]);

    double y1, y2;
    for(size_t i = 0; i < _iter - 1; ++i)
    {
        const double rate = fib_list[_iter - 1 - i] / fib_list[_iter - i];
        const double a3   = a2 - rate * (a2 - a1);
        const double a4   = a1 + rate * (a2 - a1);
        const double y3   = _func({a3}).fom();
        const double y4   = _func({a4}).fom();

        if (y3 < y4)
        {
            a2 = a4;
            y2 = y4;
        }
        else
        {
            a1 = a3;
            y1 = y3;
        }
    }
    return _func({a1});
}
GoldenSelection::GoldenSelection(ObjFunc f, double lb, double ub, size_t iter) noexcept
    : Optimizer1D(f),
      _lb(lb),
      _ub(ub),
      _iter(iter)
{}
Extrapolation::Extrapolation(ObjFunc f, Paras i, double min_len, double max_len) noexcept
    : Optimizer1D(f),
      _init(i),
      _min_len(min_len),
      _max_len(max_len)
{
    if (!(min_len > 0 && max_len > 0 && min_len < max_len))
    {
        std::cerr << "Not satisfied: min_len > 0 && max_len > 0 && min_len < max_len" << std::endl;
        exit(EXIT_FAILURE);
    }
}
Solution GoldenSelection::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    double a1 = _lb;
    double a2 = _ub;
    if (a1 > a2)
    {
        cerr << ("Range is [" + to_string(a1) + ", " + to_string(a2) + "]") << endl;
        exit(EXIT_FAILURE);
    }

    const double rate = (sqrt(5) - 1) / 2;
    double y1, y2;
    for (size_t i = _iter - 1; i > 0; --i)
    {
        const double interv_len = a2 - a1;
        const double a3 = a2 - rate * interv_len;
        const double a4 = a1 + rate * interv_len;
        if (a3 == a4)
            break;
        else
        {
            assert(a3 < a4);
            const double y3 = _func({a3}).fom();
            const double y4 = _func({a4}).fom();
            if (y3 < y4)
            {
                a2 = a4;
                y2 = y4;
            }
            else
            {
                a1 = a3;
                y1 = y3;
            }
        }
    }
    return y1 < y2 ? _func({a1}) : _func({a2});
}
Solution Extrapolation::optimize() noexcept
{
    // 1-D function
    double step = _min_len;
    double x1 = _init[0];
    double x2 = x1 + step;
    double y1 = _func({x1}).fom();
    double y2 = _func({x2}).fom();

    double lb = x1;
    double ub = x1 + _max_len;
    if (y2 > y1)
    {
        step *= -1;
        ub = x1 - _min_len;
        lb = x1 - _max_len;
        x2 = x1 + step;
        y2 = _func({x2}).fom();
        if (y2 > y1) return _func({x1});
    }
    double factor = 2;
    double x3 = x2 + factor * step;
    double y3 = _func({x3}).fom();
    double xa, xc;
    double ya, yc;
    if (y3 > y2)
    {
        xa = x1;
        xc = x3;
        ya = y1;
        yc = y3;
    }
    else
    {
        while (y3 < y2 && (lb < x3 && x3 < ub))
        {
            factor *= 2;
            x3 += factor * step;
            if (x3 >= ub) x3 = ub;
            if (x3 <= lb) x3 = lb;
            y3 = _func({x3}).fom();
        }
        double xtmp1 = x3 - factor * step;
        double xtmp2 = x3 - (factor / 2) * step;
        double ytmp1 = _func({xtmp1}).fom();
        double ytmp2 = _func({xtmp2}).fom();
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
    }

    if (xa > xc)
    {
        std::swap(xa, xc);
        std::swap(ya, yc);
    }
    const double interv_len = xc - xa;
    const size_t gso_iter = 2 + static_cast<size_t>(log10(_min_len / interv_len) / log10(0.618));
    return GoldenSelection(_func, xa, xc, gso_iter).optimize();
}
