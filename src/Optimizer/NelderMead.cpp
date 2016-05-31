#include "NelderMead.h"
using namespace std;
using namespace Eigen;
#include <iostream>
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
    for (const auto& s : _sols)
    {
        const Paras& pp = s.solution();
        assert(pp.size() == _dim);
        for (size_t i = 0; i < _dim; ++i)
        {
            if (pp[i] < min_vec[i]) min_vec[i] = pp[i];
            if (pp[i] > max_vec[i]) max_vec[i] = pp[i];
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
    for (size_t i = 0; i < _dim + 1; ++i) _sols.push_back(run_func(_inits[i]));
    double len = numeric_limits<double>::infinity();
    while (eval_counter() < _max_iter)
    {
        // 1. order
        std::sort(_sols.begin(), _sols.end(), std::less<Solution>());
        len = max_simplex_len();
        if (len < _converge_len) break;
        const Solution& worst     = _sols[_dim];
        const Solution& sec_worst = _sols[_dim - 1];
        const Solution& best      = _sols[0];

        // 2. centroid calc
        Paras centroid(_dim, 0);
        for (size_t i = 0; i < _dim; ++i) centroid = centroid + _sols[i].solution();
        centroid = 1.0 / static_cast<double>(_dim) * centroid;

        // 3. reflection
        Solution reflect = run_func(centroid + _alpha * (centroid - worst.solution()));
#ifdef WRITE_LOG
        write_log(reflect);
#endif
        if (best <= reflect && reflect < sec_worst)
        {
            _sols[_dim] = reflect;
            continue;
        }
        // 4. expansion
        else if (reflect < best)
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
            if (contracted < worst)
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
                for (size_t i = 1; i < _dim + 1; ++i)
                {
                    Paras p =
                        _sols[0].solution() - _sigma * (_sols[i].solution() - _sols[0].solution());
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
