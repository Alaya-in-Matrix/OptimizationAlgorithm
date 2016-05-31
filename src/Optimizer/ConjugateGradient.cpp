#include "ConjugateGradient.h"
using namespace std;
using namespace Eigen;
void ConjugateGradient::write_log(const Solution& s, const VectorXd& grad,
                                  const VectorXd& conj_grad) noexcept
{
    const Paras& point = s.solution();
    _log << endl;
    _log << "point: " << Map<const RowVectorXd>(&point[0], 1, _dim) << endl;
    _log << "fom:   " << s.fom() << endl;
    _log << "grad:  " << grad.transpose() << endl;
    _log << "conj_grad: " << conj_grad.transpose() << endl;
    _log << "grad_norm: " << grad.lpNorm<2>() << endl;
}
Solution ConjugateGradient::optimize() noexcept
{
    clear_counter();
    _log << _func_name << endl;

    Solution sol = run_func(_init);
    VectorXd grad = get_gradient(sol);
    VectorXd conj_grad = grad;
    double grad_norm = grad.lpNorm<2>();
    double len_walk = numeric_limits<double>::infinity();
    assert(sol.solution().size() == _dim);
    while (grad_norm > _zero_grad && eval_counter() < _max_iter && len_walk > _min_walk)
    {
        conj_grad = grad;
        for (size_t i = 0; i < _dim; ++i)
        {
            LOG(sol, grad, conj_grad);
            const Solution new_sol = run_line_search(sol, -1 * conj_grad);
            VectorXd new_grad = get_gradient(new_sol);
            double beta = pow(new_grad.lpNorm<2>() / grad.lpNorm<2>(), 2);

            len_walk = vec_norm(new_sol.solution() - sol.solution());
            sol = new_sol;
            conj_grad = new_grad + beta * conj_grad;
            grad = new_grad;
            grad_norm = grad.lpNorm<2>();
            if (!(grad_norm > _zero_grad)) break;
        }
    }
    _log << "=======================================" << endl;
    write_log(sol, grad, conj_grad);
    _log << "len_walk:    " << len_walk             << endl;
    _log << "eval:        " << eval_counter()       << endl;
    _log << "line search: " << linesearch_counter() << endl;
    if (eval_counter() >= _max_iter) 
        _log << "max iter reached" << endl;
    return sol;
}
