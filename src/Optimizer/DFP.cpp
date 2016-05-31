#include "DFP.h"
using namespace std;
using namespace Eigen;
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

    while (grad_norm > _zero_grad && eval_counter() < _max_iter && len_walk > _min_walk)
    {
#ifdef WRITE_LOG
        write_log(sol, grad, quasi_hess_inverse);
#endif
        VectorXd dvec = -1 * (quasi_hess_inverse * grad);
#ifdef WRITE_LOG
        const double judge = grad.transpose() * dvec;
        _log << "judge: " << judge << endl;
        if(judge > 0)
            _log << "judge greater than zero" << endl;
#endif
        const Solution new_sol       = run_line_search(sol, dvec);
        const VectorXd new_grad      = get_gradient(new_sol);
        const vector<double> delta_x = new_sol.solution() - sol.solution();
        const VectorXd ev_dg         = new_grad - grad;
        len_walk = vec_norm(delta_x);
        const Map<const VectorXd> ev_dx(&delta_x[0], _dim, 1);
        if (len_walk > 0)
        {
            quasi_hess_inverse +=
                (ev_dx * ev_dx.transpose()) / (ev_dx.transpose() * ev_dg) -
                (quasi_hess_inverse * ev_dg * ev_dg.transpose() * quasi_hess_inverse) /
                    (ev_dg.transpose() * quasi_hess_inverse * ev_dg);

            sol = new_sol;
            grad = new_grad;
            grad_norm = grad.lpNorm<2>();
        }
    }
    _log << "=======================================" << endl;
    write_log(sol, grad, quasi_hess_inverse);
    _log << "len_walk:    " << len_walk             << endl;
    _log << "eval:        " << eval_counter()       << endl;
    _log << "line search: " << linesearch_counter() << endl;
    if (eval_counter() >= _max_iter) 
        _log << "max iter reached" << endl;
    return sol;
}
