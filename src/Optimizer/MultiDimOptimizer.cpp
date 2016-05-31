#include "MultiDimOptimizer.h"
using namespace std;
using namespace Eigen;
MultiDimOptimizer::MultiDimOptimizer(ObjFunc f, size_t d, size_t max_iter, double min_walk,
                                     double max_walk, std::string func_name,
                                     std::string algo_name) noexcept
    : _dim(d),
      _max_iter(max_iter),
      _min_walk(min_walk),
      _max_walk(max_walk),
      _func_name(func_name),
      _algo_name(algo_name),
      _log(func_name + "." + algo_name + ".log"),
      _func(f),
      _line_searcher([&](const Paras p) -> Solution { return run_func(p); }, _log, 1e-4, 0.75),
      _eval_counter(0),
      _linesearch_counter(0)
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
Solution MultiDimOptimizer::run_line_search(const Solution& s, const VectorXd& direction) noexcept
{
    ++ _linesearch_counter;
    const Solution step_sol = _line_searcher.search(s, direction, _min_walk, _max_walk);
    const Paras& p0         = s.solution();
    const double step       = step_sol.solution()[0];
    assert(step_sol.solution().size() == 1);
    return Solution(p0 + step * direction, {0}, step_sol.fom());
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
    double fom0 = s.fom();
    for (size_t i = 0; i < _dim; ++i)
    {
        for (size_t j = i; j < _dim; ++j)
        {
            Paras p1 = p;
            p1[i] += _epsilon;
            p1[j] += _epsilon;
            double fom1 = run_func(p1).fom();
            h(i, j) = ((fom1 - fom0) / (_epsilon)-grad(i) - grad(j)) / _epsilon;
            h(j, i) = h(i, j);
        }
    }
    return h;
}
