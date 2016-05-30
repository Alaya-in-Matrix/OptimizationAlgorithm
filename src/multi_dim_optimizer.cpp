#include "multi_dim_optimizer.h"
using namespace std;
using namespace Eigen;
MultiDimOptimizer::MultiDimOptimizer(ObjFunc f, size_t d, size_t max_iter, double min_walk, double max_walk,
                      std::string func_name, std::string algo_name) noexcept
    : _func(f), 
      _dim(d), 
      _max_iter(max_iter), 
      _min_walk(min_walk), 
      _max_walk(max_walk), 
      _func_name(func_name), 
      _algo_name(algo_name), 
      _eval_counter(0), 
      _linesearch_counter(0), 
      _log(func_name + "." + algo_name + ".log")
{
    if(! _log.is_open())
    {
        cerr << "Fail to create log" << endl;
        cerr << "Func name: " << _func_name << endl << "Algo name: " << _algo_name << endl;
        exit(EXIT_FAILURE);
    }
    _log << setprecision(16);
}
Solution MultiDimOptimizer::run_func(const Paras& p) noexcept
{
    ++_eval_counter;
    return _func(p);
}
Solution MultiDimOptimizer::line_search(const Paras& point, const VectorXd& direction) noexcept
{
    ++_linesearch_counter;
    double max_step = _max_walk / direction.lpNorm<2>();
    double min_step = _min_walk / direction.lpNorm<2>();
    assert(point.size() == static_cast<size_t>(direction.size()));
    assert(max_step > min_step);

    ObjFunc line_func = [&](const vector<double> step) -> Solution {
        Paras p = point;
        const double factor = step[0];
        for (size_t i = 0; i < p.size(); ++i) p[i] += factor * direction[i];
        return run_func(p);
    };
    return Extrapolation(line_func, {0}, min_step, max_step).optimize();
}
Solution MultiDimOptimizer::armijo_bracketing_linesearch(const Paras& point, const VectorXd& direction, double guess) noexcept
{
    ++_linesearch_counter;
    const double max_step = _max_walk / direction.lpNorm<2>();
    const double min_step = _min_walk / direction.lpNorm<2>();
    const double c        = 1e-4;
    const double rho      = 0.618;
    if(guess > max_step)
        guess = max_step;
    
    ObjFunc line_func = [&](const vector<double> step) -> Solution {
        Paras p = point;
        const double factor = step[0];
        for (size_t i = 0; i < p.size(); ++i) p[i] += factor * direction[i];
        return run_func(p);
    };
    double epsi   = 1e-4;
    double step   = guess;
    Solution sol0 = line_func({0});
    double g0     = (line_func({epsi}).fom() - sol0.fom()) / epsi;
    Solution sol  = line_func({step});
    while(step > min_step && sol.fom() > sol0.fom() + c * step * g0)
    {
        step = step * rho;
        sol  = line_func({step});
    }
#ifdef WRITE_LOG
    _log << "direction: " << direction.transpose() << endl;
    _log << "max_step:  " << max_step << endl;
    _log << "min_step:  " << min_step << endl;
    _log << "guess:     " << guess << endl;
    _log << "step:      " << step  << endl;
#endif
    return sol;
}
Solution MultiDimOptimizer::interpolation(const Solution& sol, const Eigen::VectorXd& direction,
                                          double guess) noexcept
{
    ++_linesearch_counter;
    auto line_func = [&](double step) -> Solution {
        Paras p = sol.solution();
        for (size_t i = 0; i < p.size(); ++i) 
            p[i] += step * direction[i];
        return run_func(p);
    };
    const double max_step = _max_walk / direction.lpNorm<2>();
    const double min_step = _min_walk / direction.lpNorm<2>();
    if(guess > max_step) guess = max_step;
    const double c        = 1e-4;
    double g0 = (line_func(min_step).fom() - sol.fom()) / min_step;
#ifdef WRITE_LOG
    _log << "direction: " << direction.transpose() << endl;
    _log << "direction norm: " << direction.lpNorm<2>() << endl;
    _log << "max_step: " << max_step << endl;
    _log << "min_step: " << min_step << endl;
    _log << "guess:    " << guess << endl;
    _log << "g0: "       << g0 << endl;
#endif
    if(g0 > 0)
    {
        return sol;
    }
    
    double step      = guess;
    Solution new_sol = line_func(step);
    Solution last1   = new_sol;
    Solution last2   = new_sol;
    if(new_sol.fom() < sol.fom() + c * step * g0)
    {
        return new_sol;
    }
    else if(line_func(min_step).fom() > sol.fom() + c * min_step * g0)
    {
        // minimimum less than min_step
        return sol;
    }
    else
    {
        step    = (-0.5 * g0 * step * step) / (new_sol.fom() - sol.fom() - g0 * step);
#ifdef WRITE_LOG
    _log << "step: " << step << endl;
#endif
        if(step < min_step) 
            step = guess / 2;
        new_sol  = line_func(step);
        int damp = 0;
        while(damp < 2 || (new_sol.fom() > sol.fom() + c * step * g0))
        {
            ++ damp;
            last2 = last1;
            last1 = new_sol;

            const double a1 = vec_norm(last1.solution() - sol.solution()) / direction.lpNorm<2>();
            const double a2 = vec_norm(last2.solution() - sol.solution()) / direction.lpNorm<2>();
            const double y1 = last1.fom();
            const double y2 = last2.fom();
            VectorXd vec(2);
            MatrixXd mat(2, 2);
            vec(0)    = y1 - g0 * a1 - sol.fom();
            vec(1)    = y2 - g0 * a2 - sol.fom();
            mat(0, 0) = pow(a1, 3);
            mat(0, 1) = pow(a1, 2);
            mat(1, 0) = pow(a2, 3);
            mat(1, 1) = pow(a2, 2);
            VectorXd lambda_vec = mat.colPivHouseholderQr().solve(vec);
            double lambda_1 = lambda_vec(0);
            double lambda_2 = lambda_vec(1);
            
            double new_step;
            if(lambda_1 != 0)
            {
                new_step = (-1 * lambda_2 + sqrt(pow(lambda_2, 2) - 3 * lambda_1 * g0)) / (3 * lambda_1);
                if(std::isnan(new_step) || new_step < 0)
                    new_step = std::max(a1, a2);
            }
            else if(lambda_1 == 0 && lambda_2 > 0)
            {
                new_step = g0 / (-2 * lambda_2);
            }
            else
            {
                new_step = std::max(a1, a2);
            }
            if(fabs(new_step - step) < min_step || new_step < min_step)
                new_step = step / 2;
            step = new_step;
#ifdef WRITE_LOG
            // _log << "a1 = " << a1 << ", a2 = " << a2 << endl;
            // _log << "y1 = " << y1 << ", y2 = " << y2 << endl;
            // _log << last1.solution()[0] << ", " << last1.solution()[1] << endl;
            // _log << last2.solution()[0] << ", " << last2.solution()[1] << endl;
            _log << "lambda: " << lambda_vec.transpose() << ", step: " << step << endl;
#endif
            new_sol = line_func(step);
        }
        return new_sol;
    }
}
Solution MultiDimOptimizer::wolfe_linesearch(const Solution& sol,
                                             const Eigen::VectorXd& direction) noexcept
{
    ++_linesearch_counter;
    auto line_func = [&](double step) -> Solution {
        Paras p = sol.solution();
        for (size_t i = 0; i < p.size(); ++i) 
            p[i] += step * direction[i];
        return run_func(p);
    };
    const double c1       = 1e-4;
    const double c2       = 0.9;
    const double max_step = _max_walk / direction.lpNorm<2>();
    const double min_step = _min_walk / direction.lpNorm<2>();
    const double y0       = sol.fom();
    const double g0       = (line_func(min_step).fom() - sol.fom()) / min_step;
#ifdef WRITE_LOG
    _log << "direction: " << direction.transpose() << endl;
    _log << "direction norm: " << direction.lpNorm<2>() << endl;
    _log << "max_step: " << max_step << endl;
    _log << "min_step: " << min_step << endl;
    _log << "g0: "       << g0 << endl;
#endif
    if(g0 >= 0 || line_func(min_step).fom() - y0 > c1 * min_step * g0)
    {
        // even min_step is not suffciently small!
        return sol;
    }
    double step_lo   = 0;
    double step_hi   = std::min(max_step, 64 * min_step);
    double g_lo      = g0;
    Solution sol_lo  = sol;
    size_t i         = 0;
    double best_step = 0;
    while(true)
    {
#ifdef WRITE_LOG
    _log << "step_hi: " << step_hi << endl;
#endif
        Solution sol_hi = line_func(step_hi);
        if(sol_hi.fom() - y0 > c1 * step_hi * g0 || (i > 1 && sol_hi.fom() >= sol_lo.fom()))
        {
            best_step = zoom(sol, g0, direction, step_lo, step_hi);
            break;
        }
        double g_hi = (line_func(step_hi + min_step).fom() - sol_hi.fom()) / min_step;
        if(fabs(g_hi) <= -1 * c2 * g0)
        {
            best_step = step_hi;
            break;
        }
        if(g_hi >= 0)
        {
            best_step = zoom(sol, g0, direction, step_lo, step_hi);
            break;
        }
        double x1 = step_lo,  y1 = sol_lo.fom(), g1 = g_lo;
        double x2 = step_hi,  y2 = sol_hi.fom();
        MatrixXd mat(3, 3);
        mat << (2*x1), 1, 0, 
               (x1*x1), x1, 1, 
               (x2*x2), x2, 1;
        VectorXd vec(3);
        vec << g1, y1, y2;
        VectorXd abc = mat.colPivHouseholderQr().solve(vec);
        double new_step_hi = 0;
        if(abc(0) > 0)
            new_step_hi = -1 * abc(1) / (2 * abc(0));
        else
            new_step_hi = (step_hi + max_step) / 2;
#ifdef WRITE_LOG
        _log << "new_step_hi: " << new_step_hi << endl;
#endif
        if(new_step_hi / step_hi < 2.0) new_step_hi = 2 * step_hi;
        step_lo = step_hi;
        sol_lo  = sol_hi;
        g_lo    = g_hi;
        if(new_step_hi > max_step)
        {
            _log << "step_hi above max_step << endl" << endl;
            if(step_lo > (min_step + max_step) / 2)
            {
                _log << "set step_lo as best step, curvature condition maight be violated" << endl;
                cerr << "set step_lo as best step, curvature condition maight be violated" << endl;
                best_step = step_lo;
                break;
            }
            else
            {
                new_step_hi = (step_hi + max_step) / 2;
            }
        }
        step_hi = new_step_hi;
        ++i;
    }
#ifdef WRITE_LOG
    _log << "step: " << best_step << endl;
#endif
    return line_func(best_step);
}
double MultiDimOptimizer::zoom(const Solution& sol, double g0, const VectorXd& direction, double lo, double hi) noexcept
{
    auto line_func = [&](double step) -> Solution {
        Paras p = sol.solution();
        for (size_t i = 0; i < p.size(); ++i) 
            p[i] += step * direction[i];
        return run_func(p);
    };
    const double c1       = 1e-4;
    const double c2       = 0.9;
    const double min_step = _min_walk / direction.lpNorm<2>();
    double best_step      = 0;
    Solution sol_lo       = line_func(lo);
    Solution sol_hi       = line_func(hi);
    while(true)
    {
        const double   step = (lo + hi) / 2;
        if(step < min_step) 
            return 0;
#ifdef WRITE_LOG
        _log << "lo: " << lo << ", hi: " << hi << ", step: " << step << endl;
#endif
        const Solution res  = line_func(step);
        if(res.fom() > sol.fom() + c1 * step * g0 || res.fom() >= sol_lo.fom())
        {
            hi     = step;
            sol_hi = res;
        }
        else
        {
            const double g_step   = (line_func(step + min_step).fom() - res.fom()) / min_step;
#ifdef WRITE_LOG
#endif
            if(fabs(g_step) <= -1 * c2 * g0)
            {
                best_step = step;
                break;
            }
            if(g_step * (hi - lo) >= 0)
            {
                hi     = lo;
                sol_hi = sol_lo;
            }
            lo     = step;
            sol_lo = res;
        }
#ifdef WRITE_LOG
        _log << "=========================" << endl;
#endif
    }
    return best_step;
}
GradientMethod::GradientMethod(ObjFunc f, size_t d, Paras i, double epsi, double zgrad,
                               double min_walk, double max_walk, size_t max_iter, string fname,
                               string aname) noexcept
    : MultiDimOptimizer(f, d, max_iter, min_walk, max_walk, fname, aname),
      _init(i),
      _epsilon(epsi),
      _zero_grad(zgrad)
{}
VectorXd GradientMethod::get_gradient(const Solution& s) noexcept
{
    assert(_epsilon > 0);
    VectorXd grad(_dim);
    const Paras& p = s.solution();
    const double y = s.fom();
    for (size_t i = 0; i < _dim; ++i)
    {
        Paras pp = p;
        pp[i] = pp[i] + _epsilon;
        grad(i) = (run_func(pp).fom() - y) / _epsilon;
    }
    return grad;
}
MatrixXd GradientMethod::hessian(const Solution& s, const VectorXd& grad) noexcept
{
    assert(s.solution().size() == _dim);
    MatrixXd h(_dim, _dim);
    const Paras p = s.solution();
    double fom0   = s.fom();
    for (size_t i = 0; i < _dim; ++i)
    {
        for (size_t j = i; j < _dim; ++j)
        {
            Paras p1 = p;
            p1[i] += _epsilon;
            p1[j] += _epsilon;
            double fom1 = run_func(p1).fom();
            h(i, j) = ((fom1 - fom0) / (_epsilon) - grad(i) - grad(j)) / _epsilon;
            h(j, i)   = h(i, j);
        }
    }
    return h;
}
void GradientDescent::write_log(const Solution& s, const VectorXd& grad) noexcept
{
    const Paras& point = s.solution();
    _log << endl;
    _log << "point: "     << Map<const MatrixXd>(&point[0], 1, _dim) << endl;
    _log << "fom:   "     << s.fom()                                 << endl;
    _log << "grad:      " << grad.transpose()                        << endl;
    _log << "grad_norm: " << grad.lpNorm<2>()                        << endl;
}
Solution GradientDescent::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    Solution sol     = run_func(_init);
    VectorXd grad    = get_gradient(sol);
    double grad_norm = grad.lpNorm<2>();
    double len_walk  = numeric_limits<double>::infinity();
    double deltaFom  = -1 * numeric_limits<double>::infinity();
    while (grad_norm > _zero_grad && _eval_counter < _max_iter && len_walk > _min_walk)
    {
#ifdef WRITE_LOG
        write_log(sol, grad);
#endif
        const Solution new_sol   = wolfe_linesearch(sol, -1 * grad);
        deltaFom  = new_sol.fom() - sol.fom();
        len_walk  = vec_norm(new_sol.solution() - sol.solution());
        sol       = new_sol;
        grad      = get_gradient(sol);
        grad_norm = grad.lpNorm<2>();
    }
    _log << "=======================================" << endl;
    write_log(sol, grad);
    _log << "len_walk: " << len_walk << endl;
    _log << "eval:     " << _eval_counter << endl;
    _log << "line search:     " << _linesearch_counter << endl;
    if (_eval_counter >= _max_iter) _log << "max iter reached" << endl;
    return sol;
}
void ConjugateGradient::write_log(const Solution& s, const VectorXd& grad, const VectorXd& conj_grad) noexcept
{
    const Paras& point = s.solution();
    _log << endl;
    _log << "point: "     << Map<const RowVectorXd>(&point[0], 1, _dim) << endl;
    _log << "fom:   "     << s.fom() << endl;
    _log << "grad:  "     << grad.transpose() << endl;
    _log << "conj_grad: " << conj_grad.transpose() << endl;
    _log << "grad_norm: " << grad.lpNorm<2>() << endl;
}
Solution ConjugateGradient::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    Solution sol       = run_func(_init);
    VectorXd grad      = get_gradient(sol);
    VectorXd conj_grad = grad;
    double grad_norm   = grad.lpNorm<2>();
    double len_walk    = numeric_limits<double>::infinity();
    assert(sol.solution().size() == _dim);
    while (grad_norm > _zero_grad && _eval_counter < _max_iter && len_walk > _min_walk)
    {
        conj_grad = grad;
        for (size_t i = 0; i < _dim; ++i)
        {
#ifdef WRITE_LOG
            write_log(sol, grad, conj_grad);
#endif
            const Solution new_sol = wolfe_linesearch(sol, -1 * conj_grad);
            VectorXd new_grad      = get_gradient(new_sol);
            double beta            = pow(new_grad.lpNorm<2>() / grad.lpNorm<2>(), 2);

            len_walk  = vec_norm(new_sol.solution() - sol.solution());
            sol       = new_sol;
            conj_grad = new_grad + beta * conj_grad;
            grad      = new_grad;
            grad_norm = grad.lpNorm<2>();
            if (!(grad_norm > _zero_grad)) break;
        }
    }
    _log << "=======================================" << endl;
    write_log(sol, grad, conj_grad);
    _log << "len_walk: " << len_walk << endl;
    _log << "eval:     " << _eval_counter << endl;
    _log << "line search: " << _linesearch_counter << endl;
    if (_eval_counter >= _max_iter) _log << "max iter reached" << endl;
    return sol;
}

void Newton::write_log(const Solution& s, const VectorXd& grad, const MatrixXd& hess) noexcept
{
    const Paras& point = s.solution();
    _log << endl;
    _log << "point:     " << Map<const RowVectorXd>(&point[0], _dim) << endl;
    _log << "fom:       " << s.fom() << endl;
    _log << "grad:      " << grad.transpose() << endl;
    _log << "grad_norm: " << grad.lpNorm<2>() << endl;
    _log << "hessian:   " << endl << hess << endl;
}
Solution Newton::optimize() noexcept
{
    clear_counter();
    _log << "func: " << _func_name << endl;
    Solution sol       = run_func(_init);
    VectorXd grad      = get_gradient(sol);
    MatrixXd hess      = hessian(sol, grad);
    double   grad_norm = grad.lpNorm<2>();
    double   len_walk  = numeric_limits<double>::infinity();
    while (grad_norm > _zero_grad && _eval_counter < _max_iter && len_walk > _min_walk)
    {
        VectorXd direction = -1 * hess.colPivHouseholderQr().solve(grad);
        double judge       = grad.transpose() * direction;
        double dir         = judge < 0 ? 1 : -1;
#ifdef WRITE_LOG
        write_log(sol, grad, hess);
#endif
        direction *= dir;
        Solution new_sol = wolfe_linesearch(sol, direction);
        len_walk         = vec_norm(new_sol.solution() - sol.solution());
#ifdef WRITE_LOG
        _log << "len walk: " << len_walk << endl;
#endif
        sol       = new_sol;
        grad      = get_gradient(sol);
        hess      = hessian(sol, grad);
        grad_norm = grad.lpNorm<2>();

    }
    _log << "=======================================" << endl;
    write_log(sol, grad, hess);
    _log << "len_walk: " << len_walk << endl;
    _log << "iter:     " << _eval_counter << endl;
    _log << "line search:     " << _linesearch_counter << endl;
    _log << "eigenvalues of hess: " << endl << hess.eigenvalues() << endl;
    if (_eval_counter >= _max_iter) _log << "max iter reached" << endl;
    return sol;
}
void DFP::write_log(const Solution& s, const VectorXd& grad, const MatrixXd& quasi_hess) noexcept
{
    const Paras& p = s.solution();
    _log << endl;
    _log << "point:     " << Map<const RowVectorXd>(&p[0], _dim) << endl;
    _log << "fom:       " << s.fom() << endl;
    _log << "grad:      " << grad.transpose() << endl;
    _log << "grad_norm: " << grad.lpNorm<2>() << endl;
    _log << "inverse of quasi_hess:   " << endl << quasi_hess << endl;
}
Solution DFP::optimize() noexcept
{
    clear_counter();
    _log << "func: " << _func_name << endl;

    Solution sol     = run_func(_init);
    VectorXd grad    = get_gradient(sol);
    double grad_norm = grad.lpNorm<2>();
    double len_walk  = numeric_limits<double>::infinity();
    MatrixXd quasi_hess_inverse = MatrixXd::Identity(_dim, _dim);

    while (grad_norm > _zero_grad && _eval_counter < _max_iter && len_walk > _min_walk)
    {
#ifdef WRITE_LOG
        write_log(sol, grad, quasi_hess_inverse);
#endif
        VectorXd dvec = -1 * (quasi_hess_inverse * grad);
#ifdef WRITE_LOG
        const double judge = grad.transpose() * dvec;
        _log << "judge: " << judge << endl;
        assert(judge < 0);
#endif
        const Solution new_sol       = wolfe_linesearch(sol, dvec);
        const VectorXd new_grad      = get_gradient(new_sol);
        const vector<double> delta_x = new_sol.solution() - sol.solution();
        const VectorXd ev_dg         = new_grad - grad;
        len_walk                     = vec_norm(delta_x);
        const Map<const VectorXd> ev_dx(&delta_x[0], _dim, 1);
        if (len_walk > 0)
        {
            quasi_hess_inverse +=
                (ev_dx * ev_dx.transpose()) / (ev_dx.transpose() * ev_dg) -
                (quasi_hess_inverse * ev_dg * ev_dg.transpose() * quasi_hess_inverse) /
                    (ev_dg.transpose() * quasi_hess_inverse * ev_dg);

            sol       = new_sol;
            grad      = new_grad;
            grad_norm = grad.lpNorm<2>();
        }
    }
    _log << "=======================================" << endl;
    write_log(sol, grad, quasi_hess_inverse);
    _log << "len_walk: " << len_walk << endl;
    _log << "eval:     " << _eval_counter << endl;
    _log << "line search:     " << _linesearch_counter << endl;
    if (_eval_counter >= _max_iter) _log << "max iter reached" << endl;
    return sol;
}
void BFGS::write_log(const Solution& s, const VectorXd& grad, const MatrixXd& quasi_hess) noexcept
{
    const Paras& point = s.solution();
    _log << endl;
    _log << "point:     " << Map<const RowVectorXd>(&point[0], _dim) << endl;
    _log << "fom:       " << s.fom() << endl;
    _log << "grad:      " << grad.transpose() << endl;
    _log << "grad_norm: " << grad.lpNorm<2>() << endl;
    _log << "quasi_hess:   " << endl << quasi_hess << endl;
}
Solution BFGS::optimize() noexcept
{
    clear_counter();
    _log << "func: " << _func_name << endl;

    Solution sol        = run_func(_init);
    VectorXd grad       = get_gradient(sol);
    MatrixXd quasi_hess = MatrixXd::Identity(_dim, _dim);
    double   grad_norm  = grad.lpNorm<2>();
    double   len_walk   = numeric_limits<double>::infinity();

    while (grad_norm > _zero_grad && _eval_counter < _max_iter && len_walk > _min_walk)
    {
#ifdef WRITE_LOG
        write_log(sol, grad, quasi_hess);
#endif
        const VectorXd direction     = -1 * (quasi_hess.colPivHouseholderQr().solve(grad));
        const Solution new_sol       = wolfe_linesearch(sol, direction);
        const VectorXd new_grad      = get_gradient(new_sol);
        const vector<double> delta_x = new_sol.solution() - sol.solution();
        const VectorXd ev_dg         = new_grad - grad;
        const Map<const VectorXd> ev_dx(&delta_x[0], _dim, 1);
        len_walk = vec_norm(delta_x);
        if (len_walk > 0)
        {
            quasi_hess += (ev_dg * ev_dg.transpose()) / (ev_dg.transpose() * ev_dx) -
                          (quasi_hess * ev_dx * ev_dx.transpose() * quasi_hess) /
                              (ev_dx.transpose() * quasi_hess * ev_dx);

            sol       = new_sol;
            grad      = new_grad;
            grad_norm = grad.lpNorm<2>();
        }
    }
    _log << "=======================================" << endl;
    write_log(sol, grad, quasi_hess);
    _log << "len_walk: " << len_walk << endl;
    _log << "eval:     " << _eval_counter << endl;
    _log << "lien search:     " << _linesearch_counter << endl;

    if (_eval_counter >= _max_iter) _log << "max iter reached" << endl;
    return sol;
}

// MultiDimOptimizer::MultiDimOptimizer(ObjFunc f, size_t d, size_t max_iter, double min_walk, double max_walk,
//                       std::string func_name, std::string algo_name) noexcept
NelderMead::NelderMead(ObjFunc f, size_t d, std::vector<Paras> inits, double a, double g, double r,
                       double s, double conv_len, size_t max_iter, std::string fname) noexcept
    : MultiDimOptimizer(f, d, max_iter, 0, numeric_limits<double>::infinity(), fname, "NelderMead"), 
      _alpha(a),
      _gamma(g),
      _rho(r),
      _sigma(s),
      _converge_len(conv_len), 
      _inits(inits)

{
    CHECK(inits.size() == _dim + 1);
    CHECK(_alpha > 0);
    CHECK(_gamma > 0);
    CHECK(0 < _rho && _rho <= 0.5);
    CHECK(0 < _sigma && _sigma < 1);
}
double NelderMead::max_simplex_len() const noexcept
{
    const double inf = numeric_limits<double>::infinity();
    Paras min_vec(_dim, inf);
    Paras max_vec(_dim, -1 * inf);
    assert(_sols.size() == _dim + 1);
    for(const auto& s : _sols)
    {
        const Paras& pp  = s.solution();
        assert(pp.size() == _dim);
        for(size_t i = 0; i < _dim; ++i)
        {
            if(pp[i] < min_vec[i]) min_vec[i] = pp[i];
            if(pp[i] > max_vec[i]) max_vec[i] = pp[i];
        }
    }
    return vec_norm(max_vec - min_vec);
}
void NelderMead::write_log(const Solution& s) noexcept
{
    Paras p = s.solution();
    const double y = s.fom();
    _log << endl;
    _log << "point: " << Map<MatrixXd>(&p[0], 1, _dim) << endl;
    _log << "fom:   " << y << endl;
}
Solution NelderMead::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;
    _sols.clear();
    _sols.reserve(_dim + 1);
    for(size_t i = 0; i < _dim + 1; ++i)
        _sols.push_back(run_func(_inits[i]));
    double len = numeric_limits<double>::infinity();
    while(_eval_counter < _max_iter)
    {
        // 1. order
        std::sort(_sols.begin(), _sols.end(), std::less<Solution>());
        len = max_simplex_len();
        if(len < _converge_len) break;
        const Solution& worst     = _sols[_dim];
        const Solution& sec_worst = _sols[_dim - 1];
        const Solution& best      = _sols[0];

        // 2. centroid calc
        Paras centroid(_dim, 0);
        for(size_t i = 0; i < _dim; ++i)
            centroid = centroid + _sols[i].solution();
        centroid = 1.0/static_cast<double>(_dim) * centroid;

        // 3. reflection
        Solution reflect = run_func(centroid + _alpha * (centroid - worst.solution()));
#ifdef WRITE_LOG
        write_log(reflect);
#endif
        if(best <= reflect && reflect < sec_worst)
        {
            _sols[_dim] = reflect;
            continue;
        }
        // 4. expansion
        else if(reflect < best)
        {
            Solution expanded = run_func(centroid + _gamma * (reflect.solution() - centroid));
#ifdef WRITE_LOG
        write_log(expanded);
#endif
            _sols[_dim] = expanded < reflect ? expanded : reflect;
            continue;
        }
        else
        {
            // 5. contract
            assert(!(reflect < sec_worst));
            Solution contracted = run_func(centroid + _rho * (worst.solution() - centroid));
#ifdef WRITE_LOG
        write_log(contracted);
#endif
            if(contracted < worst)
            {
                _sols[_dim] = contracted;
                continue;
            }
            // 6. shrink
            else
            {
#ifdef WRITE_LOG
                _log << "shrink: " << endl;
#endif
                for(size_t i = 1; i < _dim + 1; ++i)
                {
                    Paras p  = _sols[0].solution() - _sigma * (_sols[i].solution() - _sols[0].solution());
                    _sols[i] = run_func(p);
#ifdef WRITE_LOG
                    write_log(_sols[i]);
#endif
                }
            }
        }
    }
    std::sort(_sols.begin(), _sols.end(), std::less<Solution>());
    _log << "=========================================" << endl;
    write_log(_sols[0]);
    return _sols[0];
}
Powell::Powell(ObjFunc f, size_t d, const Paras& i, size_t max_iter, double min_walk,
               double max_walk, string fname) noexcept
    : MultiDimOptimizer(f, d, max_iter, min_walk, max_walk, fname, "Powell"),
      _init(i)
{}
void Powell::write_log(const Solution& s) noexcept
{
    Paras p = s.solution();
    const double y = s.fom();
    _log << endl;
    _log << "point: " << Map<MatrixXd>(&p[0], 1, _dim) << endl;
    _log << "fom:   " << y << endl;
}
Solution Powell::optimize() noexcept
{
    clear_counter();
    Solution sol    = run_func(_init);
    double walk_len = numeric_limits<double>::infinity();

    // initial search directions are axes
    vector<VectorXd> search_direction(_dim, VectorXd(_dim));
    for(size_t i = 0; i < _dim; ++i)
    {
        for(size_t j = 0; j < _dim; ++j)
            search_direction[i][j] = i == j ? 1.0 : 0.0;
    }
    while(_eval_counter < _max_iter && walk_len > _min_walk)
    {
        double max_deltay = -1 * numeric_limits<double>::infinity();
        size_t max_delta_id;
        Paras backup_point = sol.solution();
        for(size_t i = 0; i < _dim; ++i)
        {
#ifdef WRITE_LOG
            write_log(sol);
            _log << "direction: " << search_direction[i].transpose() << endl;
#endif
            Solution search_sol = wolfe_linesearch(sol, search_direction[i]);
            if(sol.fom() - search_sol.fom() > max_deltay)
            {
                max_deltay   = sol.fom() - search_sol.fom();
                max_delta_id = i;
            }
            sol   = search_sol;
        }
        Paras new_direc                = sol.solution() - backup_point;
        VectorXd new_direc_vxd         = Map<VectorXd>(&new_direc[0], _dim);
        walk_len                       = new_direc_vxd.lpNorm<2>();
        search_direction[max_delta_id] = new_direc_vxd;
    }
    return sol;
}
