#include "def.h"
#include "optimizer.h"
#include "linear_algebra.h"
#include <random>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>
using namespace std;
using namespace Eigen;
mt19937_64 engine(RAND_SEED);

Paras Optimizer::random_init() const noexcept
{
    Paras init(_ranges.size(), 0);
    for (size_t i = 0; i < _ranges.size(); ++i)
        init[i] = uniform_real_distribution<double>(_ranges[i].first, _ranges[i].second)(engine);
    return init;
}
Solution FibOptimizer::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    if (_ranges.size() != 1) 
    {
        cerr << "FibOptimizer requires 1D function" << endl;
        exit(EXIT_FAILURE);
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if (a1 > a2) 
    {
        cerr << ("Range is [" + to_string(a1) + ", " + to_string(a2) + "]") << endl;
        exit(EXIT_FAILURE);
    }

    vector<double> fib_list{1, 1};
    if (_iter > 2)
        for (size_t i = 2; i < _iter; ++i) fib_list.push_back(fib_list[i - 1] + fib_list[i - 2]);

    double y1 = _func({a1}).fom();
    double y2 = _func({a2}).fom();
    for (size_t i = _iter - 1; i > 0; --i)
    {
        const double rate = fib_list[i - 1] / fib_list[i];

        if (y1 < y2)
        {
            a2 = a1 + rate * (a2 - a1);
            y2 = _func({a2}).fom();
        }
        else
        {
            a1 = a2 + rate * (a1 - a2);
            y1 = _func({a1}).fom();
        }
    }
    return _func({a1});
}
Solution GoldenSelection::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    if (_ranges.size() != 1)
    {
        cerr << "GoldenSelection requires 1D function" << endl;
        exit(EXIT_FAILURE);
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if (a1 > a2)
    {
        cerr << ("Range is [" + to_string(a1) + ", " + to_string(a2) + "]") << endl;
        exit(EXIT_FAILURE);
    }

    const double rate = (sqrt(5) - 1) / 2;

    double y1 = _func({a1}).fom();
    double y2 = _func({a2}).fom();
    for (size_t i = _iter - 1; i > 0; --i)
    {
        if (y1 < y2)
        {
            a2 = a1 + rate * (a2 - a1);
            y2 = _func({a2}).fom();
        }
        else
        {
            a1 = a2 + rate * (a1 - a2);
            y1 = _func({a1}).fom();
        }
    }
    return y1 < y2 ? _func({a1}) : _func({a2});
}
Solution Extrapolation::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    if (_ranges.size() != 1)
    {
        cerr << "Extrapolation requires 1D function" << endl;
        exit(EXIT_FAILURE);
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if (a1 > a2)
    {
        cerr << ("Range is [" + to_string(a1) + ", " + to_string(a2) + "]") << endl;
        exit(EXIT_FAILURE);
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
GradientMethod::GradientMethod(ObjFunc f, Range r, Paras i, double epsilon, double zgrad, size_t max_iter, string fname, string aname) noexcept
    : Optimizer(f, r, i),
      _epsilon(epsilon),
      _zero_grad(zgrad), 
      _max_iter(max_iter), 
      _func_name(fname), 
      _algo_name(aname), 
      _log(fname + "." + aname + ".log"), 
      _counter(0)
{}
vector<double> GradientMethod::get_gradient(const Paras& p) const noexcept
{
    assert(_ranges.size() == p.size());
    return get_gradient(_func, p);
}
vector<double> GradientMethod::get_gradient(ObjFunc f, const Paras& p) const noexcept
{
    const size_t dim = p.size();
    assert(_epsilon > 0);
    vector<double> grad(dim, 0);
    const double y = f(p).fom();
    for(size_t i = 0; i < dim; ++i)
    {
        Paras pp = p;
        pp[i]    = pp[i] + _epsilon;
        grad[i]  = (f(pp).fom() - y) / _epsilon;
    }
    return grad;
}
bool GradientMethod::in_range(const Paras& p) const noexcept
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
Solution GradientMethod::line_search(const Paras& point, const vector<double>& direction) const noexcept
{
    double max_step  = 2;
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
        gs_iter = 1 + log10(rate) / log10(0.618);

    if(max_step <= 0)
    {
        return _func(point);
    }
    else
    {
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
}
void GradientDescent::write_log(Paras& point, double fom, Paras& grad) noexcept
{
    const size_t dim = _ranges.size();
    if(_log.is_open())
    {
        _log << "point: "     << Map<MatrixXd>(&point[0], 1, dim)     << endl;
        _log << "fom:   "     << fom                                  << endl;
        _log << "grad:      " << Map<MatrixXd>(&grad[0], 1, dim)      << endl;
        _log << "grad_norm: " << vec_norm(grad)                       << endl << endl;
    }
}
Solution GradientDescent::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    Paras          point     = _init;
    vector<double> grad      = get_gradient(point);
    double         grad_norm = vec_norm(grad);
    Solution       best_sol  = _func(point);
    while (grad_norm > _zero_grad && in_range(point) && _counter < _max_iter)
    {
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad);
#endif
        const vector<double> direction = -1 * grad;
        // point        = line_search(point, direction).solution();
        Solution new_sol = line_search(point, direction);
        point            = new_sol.solution();
        grad             = get_gradient(point);
        grad_norm        = vec_norm(grad);
        if(new_sol < best_sol)
            best_sol = new_sol;
        ++_counter;
    }
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad);
#endif
    Paras best_point = best_sol.solution();
    if(! in_range(best_point))
        _log << "out of range" << endl;
    if(_counter >= _max_iter)
        _log << "max iter reached" << endl;
    return best_sol;
}
void ConjugateGradient::write_log(Paras& point, double fom, std::vector<double>& grad,
                                  std::vector<double>& conj_grad) noexcept
{
    const size_t dim = _ranges.size();
    if(_log.is_open())
    {
        _log << "point: "     << Map<MatrixXd>(&point[0], 1, dim)     << endl;
        _log << "fom:   "     << fom                                  << endl;
        _log << "grad:      " << Map<MatrixXd>(&grad[0], 1, dim)      << endl;
        _log << "conj_grad: " << Map<MatrixXd>(&conj_grad[0], 1, dim) << endl;
        _log << "grad_norm: " << vec_norm(grad)                       << endl << endl;
    }
}
Solution ConjugateGradient::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    const size_t dim         = _ranges.size();
    Paras point              = _init;
    vector<double> grad      = get_gradient(point);
    vector<double> conj_grad = grad;
    double         grad_norm = vec_norm(grad);
    Solution       best_sol  = _func(point);
    while(grad_norm > _zero_grad && in_range(point) && _counter < _max_iter)
    {
        for(size_t i = 0; i < dim; ++i)
        {
#ifdef WRITE_LOG
            write_log(point, _func(point).fom(), grad, conj_grad);
#endif
            const vector<double> direction = -1 * conj_grad;
            const Solution sol             = line_search(point, direction);
            const Paras new_point          = sol.solution();
            const vector<double> new_grad  = get_gradient(new_point);

            conj_grad = new_grad + pow(vec_norm(new_grad) / vec_norm(grad), 2) * conj_grad;
            grad      = new_grad;
            point     = new_point;
            grad_norm = vec_norm(grad);
            ++_counter;

            if(sol < best_sol)
                best_sol = sol;

            if(! (grad_norm > _zero_grad && in_range(point))) break;
        }
    }
#ifdef WRITE_LOG
            write_log(point, _func(point).fom(), grad, conj_grad);
#endif
    Paras best_point = best_sol.solution();
    if(! in_range(best_point))
        _log << "out of range" << endl;
    if(_counter >= _max_iter)
        _log << "max iter reached" << endl;
    return best_sol;
}

void Newton::write_log(Paras& point, double fom, std::vector<double>& grad, Eigen::MatrixXd& hess) noexcept
{
    const size_t dim = point.size();
    if(_log.is_open())
    {
        _log << "point:     " << Map<MatrixXd>(&point[0], 1, dim) << endl;
        _log << "fom:       " << fom                              << endl;
        _log << "grad:      " << Map<MatrixXd>(&grad[0], 1, dim)  << endl;
        _log << "grad_norm: " << vec_norm(grad)                   << endl;
        _log << "hessian: "   << endl << hess << endl << endl;
    }
}
Solution Newton::optimize() noexcept
{
    clear_counter();

    _log << "func: " << _func_name << endl;
    const size_t dim       = _ranges.size();
    Paras point            = _init;
    vector<double> grad    = get_gradient(point);
    MatrixXd hess          = hessian(point);
    double grad_norm       = vec_norm(grad);
    Solution best_sol      = _func(point);
    while(grad_norm > _zero_grad && in_range(point) && _counter < _max_iter)
    {
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad, hess);
#endif
        VectorXd neg_g(dim);
        for(size_t i = 0; i < dim; ++i)
            neg_g(i) = -1 * grad[i];

        VectorXd delta = hess.colPivHouseholderQr().solve(neg_g);

        vector<double> direction(dim, 0);
        for(size_t i = 0; i < dim; ++i)
            direction[i] = delta(i);

        point     = line_search(point, direction).solution();
        grad      = get_gradient(point);
        hess      = hessian(point);
        grad_norm = vec_norm(grad);

        Solution tmp_sol = _func(point);
        if(tmp_sol < best_sol)
            best_sol = tmp_sol;

        ++_counter;
    }
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad, hess);
#endif

    Paras best_point = best_sol.solution();
    if(! in_range(best_point))
        _log << "out of range" << endl;
    if(_counter >= _max_iter)
        _log << "max iter reached" << endl;
    return best_sol;
}
MatrixXd Newton::hessian(const Paras& p) const noexcept
{
    const size_t dim = _ranges.size();
    assert(p.size() == dim);
    MatrixXd h(dim, dim);
    
    for(size_t i = 0; i < dim; ++i)
    {
        ObjFunc partial_grad = [&](const Paras& p)->Solution{
            Paras pp = p;
            pp[i] += _epsilon;
            double grad = (_func(pp).fom() - _func(p).fom()) / _epsilon;
            return Solution(p, {0}, grad);
        };
        vector<double> sec_grad = get_gradient(partial_grad, p);
        for(size_t j = 0; j < dim; ++j)
        {
            h(i, j) = sec_grad[j];
        }
    }
    return h;
}
