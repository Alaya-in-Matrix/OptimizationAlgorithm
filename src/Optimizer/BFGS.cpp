#include "BFGS.h"
using namespace std;
using namespace Eigen;
void BFGS::write_log(const Solution& s, const VectorXd& grad, const MatrixXd& quasi_hess) noexcept
{
    const Paras& point = s.solution();
    _log << endl;
    _log << "point:     " << Map<const RowVectorXd>(&point[0], _dim) << endl;
    _log << "fom:       " << s.fom() << endl;
    _log << "grad:      " << grad.transpose() << endl;
    _log << "grad_norm: " << grad.lpNorm<2>() << endl;
    _log << "quasi_hess:" << endl << quasi_hess << endl;
}
Solution BFGS::optimize() noexcept
{
    clear_counter();
    _log << "func: " << _func_name << endl;

    Solution sol = run_func(_init);
    VectorXd grad = get_gradient(sol);
    MatrixXd quasi_hess = MatrixXd::Identity(_dim, _dim);
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();

    while (grad_norm > _zero_grad && eval_counter() < _max_iter && len_walk > _min_walk)
    {
#ifdef WRITE_LOG
        write_log(sol, grad, quasi_hess);
#endif
        const VectorXd direction     = -1 * (quasi_hess.colPivHouseholderQr().solve(grad));
        const Solution new_sol       = run_line_search(sol, direction);
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

            sol = new_sol;
            grad = new_grad;
            grad_norm = grad.lpNorm<2>();
        }
    }
    _log << "=======================================" << endl;
    write_log(sol, grad, quasi_hess);
    _log << "len_walk:    " << len_walk << endl;
    _log << "eval:        " << eval_counter() << endl;
    _log << "line search: " << linesearch_counter() << endl;

    if (eval_counter() >= _max_iter) 
        _log << "max iter reached" << endl;
    return sol;
}
