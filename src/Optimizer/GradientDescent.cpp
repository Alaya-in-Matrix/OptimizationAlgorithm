#include "GradientDescent.h"
using namespace std;
using namespace Eigen;
void GradientDescent::write_log(const Solution& s, const VectorXd& grad) noexcept
{
    const Paras& point = s.solution();
    _log << endl;
    _log << "point: " << Map<const MatrixXd>(&point[0], 1, _dim) << endl;
    _log << "fom:   " << s.fom() << endl;
    _log << "grad:      " << grad.transpose() << endl;
    _log << "grad_norm: " << grad.lpNorm<2>() << endl;
}
Solution GradientDescent::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    Solution sol       = run_func(_init);
    VectorXd grad      = get_gradient(sol);
    double   grad_norm = grad.lpNorm<2>();
    double   len_walk  = numeric_limits<double>::infinity();
    while (grad_norm > _zero_grad && eval_counter() < _max_iter && len_walk > _min_walk)
    {
        LOG(sol, grad);
        const Solution new_sol = run_line_search(sol, -1 * grad);
        len_walk  = vec_norm(new_sol.solution() - sol.solution());
        sol       = new_sol;
        grad      = get_gradient(sol);
        grad_norm = grad.lpNorm<2>();
    }
    _log << "=======================================" << endl;
    write_log(sol, grad);
    _log << "len_walk:    " << len_walk << endl;
    _log << "eval:        " << eval_counter() << endl;
    _log << "line search: " << linesearch_counter() << endl;
    if (eval_counter() >= _max_iter) 
        _log << "max iter reached" << endl;
    return sol;
}
