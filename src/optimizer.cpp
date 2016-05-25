#include "optimizer.h"
#include <algorithm>
using namespace std;
using namespace Eigen;
mt19937_64 engine(RAND_SEED);

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
        for (size_t i = 2; i < _iter; ++i) fib_list.push_back(fib_list[i - 1] + fib_list[i - 2]);

    double y1, y2;
    for (size_t i = _iter - 1; i > 0; --i)
    {
        const double rate = fib_list[i - 1] / fib_list[i];
        const double interv_len = a2 - a1;
        const double a3 = a2 - rate * interv_len;
        const double a4 = a1 + rate * interv_len;
        assert(a3 <= a4);
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
    return _func({a1});
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
        while (y3 < y2 && (lb <= x3 && x3 <= ub))
        {
            factor *= 2;
            x3 += factor * step;
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
    // GoldenSelection gso(_func, xa, xc, gso_iter);
    // return gso.optimize();
}
GradientMethod::GradientMethod(ObjFunc f, size_t d, Paras i, double epsi, double zgrad,
                               double min_walk, double _max_walk, size_t max_iter, string fname,
                               string aname) noexcept : Optimizer(f, d),
                                                        _init(i),
                                                        _epsilon(epsi),
                                                        _zero_grad(zgrad),
                                                        _min_walk(min_walk),
                                                        _max_walk(_max_walk),
                                                        _max_iter(max_iter),
                                                        _func_name(fname),
                                                        _algo_name(aname),
                                                        _log(fname + "." + aname + ".log"),
                                                        _counter(0)
{
    _log << setprecision(9);
}
VectorXd GradientMethod::get_gradient(const Paras& p) const noexcept
{
    assert(p.size() == _dim);
    return get_gradient(_func, p);
}
VectorXd GradientMethod::get_gradient(ObjFunc f, const Paras& p) const noexcept
{
    assert(_epsilon > 0);
    VectorXd grad(_dim);
    const double y = f(p).fom();
    for (size_t i = 0; i < _dim; ++i)
    {
        Paras pp = p;
        pp[i] = pp[i] + _epsilon;
        grad(i) = (f(pp).fom() - y) / _epsilon;
    }
    return grad;
}
MatrixXd GradientMethod::hessian(const Paras& p) const noexcept
{
    assert(p.size() == _dim);
    MatrixXd h(_dim, _dim);

    double fom0 = _func(p).fom();
    vector<double> fom1d(_dim, 0);
    for (size_t i = 0; i < _dim; ++i)
    {
        Paras pp = p;
        pp[i] += _epsilon;
        fom1d[i] = _func(pp).fom();
    }
    for (size_t i = 0; i < _dim; ++i)
    {
        for (size_t j = i; j < _dim; ++j)
        {
            Paras p1 = p;
            p1[i] += _epsilon;
            p1[j] += _epsilon;

            double fom1 = _func(p1).fom();
            double grad = (fom1 - fom1d[i] - fom1d[j] + fom0) / (_epsilon * _epsilon);
            h(i, j) = grad;
            h(j, i) = grad;
        }
    }
    return h;
}
Solution GradientMethod::line_search(const Paras& point, const VectorXd& direction) const noexcept
{
    double max_step = _max_walk / direction.lpNorm<2>();
    double min_step = _min_walk / direction.lpNorm<2>();
    assert(point.size() == _dim && static_cast<size_t>(direction.size()) == _dim);
    assert(max_step > min_step);

    ObjFunc line_func = [&](const vector<double> step) -> Solution {
        Paras p = point;
        const double factor = step[0];
        for (size_t i = 0; i < p.size(); ++i) p[i] += factor * direction[i];
        return _func(p);
    };
    return Extrapolation(line_func, {0}, min_step, max_step).optimize();
}
void GradientDescent::write_log(Paras& point, double fom, VectorXd& grad) noexcept
{
    if (_log.is_open())
    {
        _log << endl;
        _log << "point: " << Map<MatrixXd>(&point[0], 1, _dim) << endl;
        _log << "fom:   " << fom << endl;
        _log << "grad:      " << grad.transpose() << endl;
        _log << "grad_norm: " << grad.lpNorm<2>() << endl;
    }
}
Solution GradientDescent::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    Paras point = _init;
    VectorXd grad = get_gradient(point);
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();
    while (grad_norm > _zero_grad && _counter < _max_iter && len_walk > _min_walk)
    {
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad);
#endif
        const VectorXd direction = -1 * grad;
        const Solution new_sol = line_search(point, direction);

        len_walk = vec_norm(new_sol.solution() - point);
        point = new_sol.solution();
        grad = get_gradient(point);
        grad_norm = grad.lpNorm<2>();
        ++_counter;
    }
    _log << "=======================================" << endl;
    write_log(point, _func(point).fom(), grad);
    _log << "len_walk: " << len_walk << endl;
    _log << "iter:     " << _counter << endl;
    if (_counter >= _max_iter) _log << "max iter reached" << endl;
    return _func(point);
}
void ConjugateGradient::write_log(Paras& point, double fom, VectorXd& grad,
                                  VectorXd& conj_grad) noexcept
{
    if (_log.is_open())
    {
        _log << endl;
        _log << "point: " << Map<MatrixXd>(&point[0], 1, _dim) << endl;
        _log << "fom:   " << fom << endl;
        _log << "grad:  " << grad.transpose() << endl;
        _log << "conj_grad: " << conj_grad.transpose() << endl;
        _log << "grad_norm: " << grad.lpNorm<2>() << endl;
    }
}
Solution ConjugateGradient::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    Paras point = _init;
    VectorXd grad = get_gradient(point);
    VectorXd conj_grad = grad;
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();
    assert(point.size() == _dim);
    while (grad_norm > _zero_grad && _counter < _max_iter && len_walk > _min_walk)
    {
        conj_grad = grad;
        for (size_t i = 0; i < _dim; ++i)
        {
            ++_counter;
#ifdef WRITE_LOG
            write_log(point, _func(point).fom(), grad, conj_grad);
#endif
            const Solution sol = line_search(point, -1 * conj_grad);
            const Paras new_point = sol.solution();
            VectorXd new_grad = get_gradient(sol.solution());
            double beta = static_cast<double>(new_grad.transpose() * new_grad) /
                          static_cast<double>(grad.transpose() * grad);
            len_walk = vec_norm(new_point - point);
            point = new_point;
            conj_grad = new_grad + beta * conj_grad;
            grad = new_grad;
            grad_norm = grad.lpNorm<2>();
            if (!(grad_norm > _zero_grad)) break;
        }
    }
    _log << "=======================================" << endl;
    write_log(point, _func(point).fom(), grad, conj_grad);
    _log << "len_walk: " << len_walk << endl;
    _log << "iter:     " << _counter << endl;
    if (_counter >= _max_iter) _log << "max iter reached" << endl;
    return _func(point);
}

void Newton::write_log(Paras& point, double fom, VectorXd& grad, Eigen::MatrixXd& hess) noexcept
{
    // const size_t dim = point.size();
    if (_log.is_open())
    {
        VectorXd delta = hess.colPivHouseholderQr().solve(-1 * grad);
        double f1 = grad.transpose() * delta;
        double f2 = 0.5 * delta.transpose() * hess * delta;
        _log << endl;
        _log << "point:     " << Map<MatrixXd>(&point[0], 1, _dim) << endl;
        _log << "fom:       " << fom << endl;
        _log << "grad:      " << grad.transpose() << endl;
        _log << "grad_norm: " << grad.lpNorm<2>() << endl;
        _log << "hessian:   " << endl << hess << endl;
        _log << "direction: " << delta.transpose() << endl;
        _log << "judge:     " << (f1 + f2) << endl;
    }
}
Solution Newton::optimize() noexcept
{
    clear_counter();
    _log << "func: " << _func_name << endl;
    Paras point = _init;
    VectorXd grad = get_gradient(point);
    MatrixXd hess = hessian(point);
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();
    while (grad_norm > _zero_grad && _counter < _max_iter && len_walk > _min_walk)
    {
        VectorXd direction = -1 * hess.colPivHouseholderQr().solve(grad);
        double judge = grad.transpose() * direction;
        double dir = judge < 0 ? 1 : -1;
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad, hess);
#endif
        direction *= dir;
        Solution sol = line_search(point, direction);
        len_walk = vec_norm(sol.solution() - point);
#ifdef WRITE_LOG
        _log << "len walk: " << len_walk << endl;
#endif
        point = sol.solution();
        grad = get_gradient(point);
        hess = hessian(point);
        grad_norm = grad.lpNorm<2>();

        ++_counter;
    }
    _log << "=======================================" << endl;
    write_log(point, _func(point).fom(), grad, hess);
    _log << "len_walk: " << len_walk << endl;
    _log << "iter:     " << _counter << endl;
    _log << "eigenvalues of hess: " << endl << hess.eigenvalues() << endl;

    if (_counter >= _max_iter) _log << "max iter reached" << endl;
    return _func(point);
}
void DFP::write_log(Paras& p, double fom, VectorXd& grad, Eigen::MatrixXd& quasi_hess) noexcept
{
    const size_t dim = p.size();
    if (_log.is_open())
    {
        _log << endl;
        _log << "point:     " << Map<MatrixXd>(&p[0], 1, dim) << endl;
        _log << "fom:       " << fom << endl;
        _log << "grad:      " << grad.transpose() << endl;
        _log << "grad_norm: " << grad.lpNorm<2>() << endl;
        _log << "inverse of quasi_hess:   " << endl << quasi_hess << endl;
    }
}
Solution DFP::optimize() noexcept
{
    clear_counter();
    _log << "func: " << _func_name << endl;

    Paras point = _init;
    VectorXd grad = get_gradient(point);
    MatrixXd quasi_hess_inverse = MatrixXd::Identity(_dim, _dim);
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();

    while (grad_norm > _zero_grad && _counter < _max_iter && len_walk > _min_walk)
    {
        ++_counter;
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad, quasi_hess_inverse);
#endif
        VectorXd dvec = -1 * (quasi_hess_inverse * grad);

        Solution sol = line_search(point, dvec);
        const VectorXd new_grad = get_gradient(sol.solution());
        const vector<double> delta_x = sol.solution() - point;
        const VectorXd ev_dg = new_grad - grad;
        const Map<const VectorXd> ev_dx(&delta_x[0], _dim, 1);
        len_walk = vec_norm(delta_x);
        if (len_walk > 0)
        {
            quasi_hess_inverse +=
                (ev_dx * ev_dx.transpose()) / (ev_dx.transpose() * ev_dg) -
                (quasi_hess_inverse * ev_dg * ev_dg.transpose() * quasi_hess_inverse) /
                    (ev_dg.transpose() * quasi_hess_inverse * ev_dg);

            point = sol.solution();
            grad = new_grad;
            grad_norm = grad.lpNorm<2>();
        }
    }
    _log << "=======================================" << endl;
    write_log(point, _func(point).fom(), grad, quasi_hess_inverse);
    _log << "len_walk: " << len_walk << endl;
    _log << "iter:     " << _counter << endl;

    if (_counter >= _max_iter) _log << "max iter reached" << endl;

    return _func(point);
}
Solution BFGS::optimize() noexcept
{
    clear_counter();
    _log << "func: " << _func_name << endl;

    Paras point = _init;
    VectorXd grad = get_gradient(point);
    MatrixXd quasi_hess = MatrixXd::Identity(_dim, _dim);
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();

    while (grad_norm > _zero_grad && _counter < _max_iter && len_walk > _min_walk)
    {
        ++_counter;
#ifdef WRITE_LOG
        write_log(point, _func(point).fom(), grad, quasi_hess);
#endif
        VectorXd direction = -1 * (quasi_hess.colPivHouseholderQr().solve(grad));

        Solution sol = line_search(point, direction);
        const VectorXd new_grad = get_gradient(sol.solution());
        VectorXd ev_dg = new_grad - grad;
        const vector<double> delta_x = sol.solution() - point;
        const Map<const VectorXd> ev_dx(&delta_x[0], _dim, 1);
        len_walk = vec_norm(delta_x);
        if (len_walk > 0)
        {
            quasi_hess += (ev_dg * ev_dg.transpose()) / (ev_dg.transpose() * ev_dx) -
                          (quasi_hess * ev_dx * ev_dx.transpose() * quasi_hess) /
                              (ev_dx.transpose() * quasi_hess * ev_dx);

            point = sol.solution();
            grad = new_grad;
            grad_norm = grad.lpNorm<2>();
        }
    }
    _log << "=======================================" << endl;
    write_log(point, _func(point).fom(), grad, quasi_hess);
    _log << "len_walk: " << len_walk << endl;
    _log << "iter:     " << _counter << endl;

    if (_counter >= _max_iter) _log << "max iter reached" << endl;

    return _func(point);
}
void BFGS::write_log(Paras& p, double fom, VectorXd& grad, Eigen::MatrixXd& quasi_hess) noexcept
{
    const size_t dim = p.size();
    if (_log.is_open())
    {
        VectorXd gradvec = Map<VectorXd>(&grad[0], dim, 1);
        _log << endl;
        _log << "point:     " << Map<MatrixXd>(&p[0], 1, dim) << endl;
        _log << "fom:       " << fom << endl;
        _log << "grad:      " << grad.transpose() << endl;
        _log << "grad_norm: " << grad.lpNorm<2>() << endl;  
        _log << "quasi_hess:   " << endl << quasi_hess << endl;
    }
}

NelderMead::NelderMead(ObjFunc f, size_t d, std::vector<Paras> inits, double a, double g, double r,
                       double s, double conv_len, size_t max_iter, std::string fname) noexcept
    : Optimizer(f, d),
      _alpha(a),
      _gamma(g),
      _rho(r),
      _sigma(s),
      _converge_len(conv_len),
      _max_iter(max_iter), 
      _func_name(fname), 
      _log(fname + ".NelderMead.log")

{
    assert(inits.size() == _dim + 1);
    assert(_alpha > 0);
    assert(_gamma > 0);
    assert(0 < _rho && _rho <= 0.5);
    assert(0 < _sigma && _sigma < 1);
    _points.reserve(_dim + 1);
    for(size_t i = 0; i < _dim + 1; ++i)
        _points.push_back(_func(inits[i]));
}
bool NelderMead::converged() const noexcept
{
    return false;
}
Solution NelderMead::optimize() noexcept
{
    _counter = 0;
    while(_counter < _max_iter && ! converged())
    {
        // 1. order
        ++_counter;
        std::sort(_points.begin(), _points.end(), std::less<Solution>());
        const Solution& worst     = _points[_dim];
        const Solution& sec_worst = _points[_dim - 1];
        const Solution& best      = _points[0];

        // 2. centroid calc
        Paras centroid(_dim, 0);
        for(size_t i = 0; i < _dim; ++i)
            centroid = centroid + _points[i].solution();
        centroid = 1.0/static_cast<double>(_dim) * centroid;

        // 3. reflection
        Solution reflect = _func(centroid + _alpha * (centroid - worst.solution()));
        if(best <= reflect && reflect < sec_worst)
        {
            _points[_dim] = reflect;
            continue;
        }
        // 4. expansion
        else if(reflect < best)
        {
            Solution expanded = _func(centroid + _gamma * (reflect.solution() - centroid));
            _points[_dim] = expanded < reflect ? expanded : reflect;
            continue;
        }
        else
        {
            // 5. contract
            assert(!(reflect < sec_worst));
            Solution contracted = _func(centroid + _rho * (worst.solution() - centroid));
            if(contracted < worst)
            {
                _points[_dim] = contracted;
                continue;
            }
            // 6. shrink
            else
            {
                for(size_t i = 1; i < _dim + 1; ++i)
                {
                    // _points[i] = _points[0] + _sigma * (_points[i] - _points[0]);
                    Paras p = _points[0].solution() - _sigma * (_points[i].solution() - _points[0].solution());
                    _points[i] = _func(p);
                }
            }
        }
    }
    std::sort(_points.begin(), _points.end(), std::less<Solution>());
    return _points[0];
}
