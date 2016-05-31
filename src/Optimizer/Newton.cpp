#include "Newton.h"
using namespace std;
using namespace Eigen;
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
    Solution sol = run_func(_init);
    VectorXd grad = get_gradient(sol);
    MatrixXd hess = hessian(sol, grad);
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();
    while (grad_norm > _zero_grad && eval_counter() < _max_iter && len_walk > _min_walk)
    {
        VectorXd direction = -1 * hess.colPivHouseholderQr().solve(grad);
        double judge = grad.transpose() * direction;
        double dir = judge < 0 ? 1 : -1;
        LOG(sol, grad, hess);
        direction *= dir;
        Solution new_sol = run_line_search(sol, direction);
        len_walk = vec_norm(new_sol.solution() - sol.solution());
#ifdef WRITE_LOG
        _log << "len walk: " << len_walk << endl;
#endif
        sol = new_sol;
        grad = get_gradient(sol);
        hess = hessian(sol, grad);
        grad_norm = grad.lpNorm<2>();
    }
    _log << "=======================================" << endl;
    write_log(sol, grad, hess);
    _log << "len_walk:    " << len_walk             << endl;
    _log << "iter:        " << eval_counter()       << endl;
    _log << "line search: " << linesearch_counter() << endl;
    _log << "eigenvalues of hess: " << endl << hess.eigenvalues() << endl;
    if (eval_counter() >= _max_iter) 
        _log << "max iter reached" << endl;
    return sol;
}
