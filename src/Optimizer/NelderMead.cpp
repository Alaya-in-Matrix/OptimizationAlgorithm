#include "NelderMead.h"
using namespace std;
using namespace Eigen;
#define CHECK(COND)                                                              \
    {                                                                            \
        if (!(COND))                                                             \
        {                                                                        \
            std::cerr << "NelderMead initialize failed: " << #COND << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }
NelderMead::NelderMead(ObjFunc f, size_t d, std::vector<Paras> inits, double a, double g, double r,
                       double s, double conv_len, size_t max_iter, std::string fname) noexcept
    : MultiDimOptimizer(f, d, max_iter, 0, numeric_limits<double>::infinity(), fname, "NelderMead"),
      _alpha(a),
      _gamma(g),
      _rho(r),
      _sigma(s),
      _min_walk(conv_len),
      _inits(inits)

{
    CHECK(inits.size() == _dim + 1);
    CHECK(_alpha > 0);
    CHECK(_gamma > 0);
    CHECK(0 < _rho && _rho <= 0.5);
    CHECK(0 < _sigma && _sigma < 1);
}
double NelderMead::update_sols(size_t idx, const Solution& new_sol) noexcept
{
    assert(_sols.size() == _dim + 1);
    assert(idx <= _dim);
    const double walk_len = vec_norm(new_sol.solution() - _sols[idx].solution());
    _sols[idx] = new_sol;
    return walk_len;
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
    for (size_t i = 0; i < _dim + 1; ++i) _sols.push_back(run_func(_inits[i]));
    double walk_len = numeric_limits<double>::infinity();
    while (eval_counter() < _max_iter && walk_len > _min_walk)
    {
        // 1. order
        std::sort(_sols.begin(), _sols.end(), std::less<Solution>());
        const Solution& worst = _sols[_dim];
        const Solution& sec_worst = _sols[_dim - 1];
        const Solution& best = _sols[0];

        // 2. centroid calc
        Paras centroid(_dim, 0);
        for (size_t i = 0; i < _dim; ++i) centroid = centroid + _sols[i].solution();
        centroid = 1.0 / static_cast<double>(_dim) * centroid;

        // 3. reflection
        Solution reflect = run_func(centroid + _alpha * (centroid - worst.solution()));
        LOG(reflect);
        if (best <= reflect && reflect < sec_worst)
        {
            walk_len = update_sols(_dim, reflect);
        }
        else if (reflect < best)  // 4. expansion
        {
            Solution expanded = run_func(centroid + _gamma * (reflect.solution() - centroid));
            LOG(expanded);
            const Solution& new_sol = expanded < reflect ? expanded : reflect;
            walk_len = update_sols(_dim, new_sol);
        }
        else  // 5. contract
        {
            assert(!(reflect < sec_worst));
            Solution contracted = run_func(centroid + _rho * (worst.solution() - centroid));
            LOG(contracted);
            if (contracted < worst)
            {
                walk_len = update_sols(_dim, contracted);
            }
            else  // 6. shrink
            {
#ifdef WRITE_LOG
                _log << "shrink: " << endl;
#endif
                walk_len = 0;
                for (size_t i = 1; i < _dim + 1; ++i)
                {
                    Paras p         = best.solution() - _sigma * (_sols[i].solution() - best.solution());
                    double tmp_walk = update_sols(i, run_func(p));
                    walk_len        = max(tmp_walk, walk_len);
                    LOG(_sols[i]);
                }
            }
        }
    }
    std::sort(_sols.begin(), _sols.end(), std::less<Solution>());
    _log << "=========================================" << endl;
    write_log(_sols[0]);
    return _sols[0];
}
