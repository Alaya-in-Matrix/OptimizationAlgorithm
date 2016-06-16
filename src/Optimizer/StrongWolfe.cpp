#include "StrongWolfe.h"
using namespace std;
using namespace Eigen;
StrongWolfe::StrongWolfe(ObjFunc f, ofstream& l, double c1, double c2) noexcept
    : LineSearch(f, l), 
      _c1(c1), 
      _c2(c2)
{}
VectorXd StrongWolfe::cubic_interpolation(double x1, double y1, double g1, double x2, double y2,
                                          double g2, bool cubic) const noexcept
{
    VectorXd abcd(4);
    if(cubic) 
    {
        MatrixXd mat(4, 4);
        VectorXd vec(4);
        mat <<  pow(x1, 3), x1*x1, x1, 1, 
                pow(x2, 3), x2*x2, x2, 1,
                3*x1*x1,    2*x1,  1,  0,
                3*x2*x2,    2*x2,  1,  0;
        vec << y1, y2, g1, g2;
        abcd = mat.colPivHouseholderQr().solve(vec);
    }
    else  // use quad interpolation instead of cubic interpolation
    {
        MatrixXd mat(3, 3);
        VectorXd vec(3);
        mat << x1 * x1, x1, 1, 
               x2 * x2, x2, 1, 
               2  * x2, 1,  0;
        vec << y1, y2, g2;
        VectorXd abc = mat.colPivHouseholderQr().solve(vec);
        abcd << 0, abc(0), abc(1), abc(2);
    }
    return abcd;
}
Solution StrongWolfe::search(const Solution& sol, const VectorXd& direction, double min_walk,
                             double max_walk) const noexcept
{
    ObjFunc line_func = [&](Paras step) -> Solution {
        Paras p = sol.solution();
        for (size_t i = 0; i < p.size(); ++i) p[i] += step[0] * direction[i];
        // here, _func(p).solution() != step
        Solution s = _func(p);
        return Solution(step, {0}, s.fom());
    };
    const Solution zero_sol = Solution({0}, {0}, sol.fom());
    const double   min_step = min_walk / direction.lpNorm<2>();
    const double   max_step = max_walk / direction.lpNorm<2>();
    const double   y0       = zero_sol.fom();
    const double   g0       = line_grad(line_func, zero_sol, min_step);
#ifdef WRITE_LOG
        _log << "StrongWolfe Search" << endl;
        _log << "\tdirection:      " << direction.transpose() << endl;
        _log << "\tdirection norm: " << direction.lpNorm<2>() << endl;
        _log << "\tmin_step:       " << min_step              << endl;
        _log << "\tmax_step:       " << max_step              << endl;
        _log << "\ty0:             " << y0                    << endl;
        _log << "\tg0:             " << g0                    << endl;
#endif
    if(g0 >= 0) return zero_sol;

    double   step_lo  = 0;
    double   step_hi  = std::min(max_step, 64 * min_step);
    Solution sol_lo   = zero_sol;
    Solution sol_hi   = line_func({step_hi});
    double   g_lo     = g0;
    Solution best_sol = sol_lo;
    while(true)
    {
#ifdef WRITE_LOG
        _log << "\ttrial step: " << step_hi << endl;
#endif
        if(sol_hi.fom() - y0 > _c1 * step_hi * g0 || (sol_hi.fom() >= sol_lo.fom()))
        {
            best_sol = zoom(line_func, y0, g0, sol_lo, g_lo, sol_hi, min_step);
            break;
        }
        
        // step_hi satisfies sufficiently-decrease condition
        double g_hi = line_grad(line_func, sol_hi, min_step);
        if(fabs(g_hi) <= -1 * _c2 * g0) // curvature condition satisfied
        {
            best_sol = sol_hi;
            break;
        }
        if(g_hi >= 0)
        {
            best_sol = zoom(line_func, y0, g0, sol_lo, g_lo, sol_hi, g_hi, min_step);
            break;
        }

        double new_step_hi = cubic_predict(step_lo, sol_lo.fom(), g_lo, step_hi, sol_hi.fom(), g_hi, false);
        if (std::isnan(new_step_hi) || new_step_hi < 2 * step_hi) 
            new_step_hi = 2 * step_hi;
        if(new_step_hi > max_step)
        {
            if(step_hi > (min_step + max_step) / 2)
            {
#ifdef WRITE_LOG
                _log << "\tWARN: new trial step exceeds max_step, "
                     << "curvature condition might be violated" 
                     << endl;
#endif
                best_sol = sol_hi;
                break;
            }
            else
                new_step_hi = (step_hi + max_step) / 2;
        }
        step_lo = step_hi;
        sol_lo  = sol_hi;
        g_lo    = g_hi;
        step_hi = new_step_hi;
        sol_hi  = line_func({step_hi});
    }
#ifdef WRITE_LOG
    _log << "final step: " << best_sol.solution()[0] << endl;
    _log << "walk len: "   << best_sol.solution()[0] * direction.lpNorm<2>() << endl;
#endif
    return best_sol;
}

double StrongWolfe::cubic_predict(double x1, double y1, double g1, double x2, double y2,
                                  double g2, bool cubic) const noexcept
{
    // cubic_interpolation and predict the minima
    VectorXd abcd = cubic_interpolation(x1, y1, g1, x2, y2, g2, cubic);
    double a      = 3 * abcd(0);
    double b      = 2 * abcd(1);
    double c      = 1 * abcd(2);
    if(a != 0 && pow(b, 2) > 4 * a * c)
        return (-1 * b + sqrt(b*b - 4 * a * c)) / (2*a);
    else if(a == 0 && b > 0)
        return -1 * (c / b);
    else 
        return numeric_limits<double>::quiet_NaN();
}
Solution StrongWolfe::zoom(ObjFunc line_func, double y0, double g0, const Solution& sol_lo,
                           double g_lo, const Solution& sol_hi, double g_hi, double min_step) const
    noexcept
{
#ifdef WRITE_LOG
    _log << "\tzoom lo: " << sol_lo.solution()[0] << endl;
    _log << "\tzoom hi: " << sol_hi.solution()[0] << endl;
#endif
    Solution slo = sol_lo;
    Solution shi = sol_hi;
    double step  = slo.solution()[0];
    while (true)
    {
        const double step_lo  = slo.solution()[0];
        const double step_hi  = shi.solution()[0];
        const double y_lo     = slo.fom();
        const double y_hi     = shi.fom();
        double new_step = cubic_predict(step_lo, y_lo, g_lo, step_hi, y_hi, g_hi);
        
        // safe guard
        if (std::isnan(new_step) || new_step > max(step_lo, step_hi) 
                                 || new_step < min(step_hi, step_lo) 
                                 || fabs(new_step - step) <= min_step)
            new_step = (step_lo + step_hi) / 2;
        step = new_step;
        if(fabs(step - step_lo) <= min_step)
            return slo;
        else if(fabs(step - step_hi) <= min_step)
            return shi;
#ifdef WRITE_LOG
        _log << "\tzoom trial step: " << step << endl;
#endif
        Solution res = line_func({step});
        if (res.fom() > y0 + _c1 * step * g0 || res.fom() >= slo.fom())
            shi = res;
        else
        {
            const double g_step = line_grad(line_func, res, min_step);
            if (fabs(g_step) <= -1 * _c2 * g0) 
                return res;
            if (g_step * (step_hi - step_lo) >= 0) 
                shi = slo;
            slo = res;
        }
    }
}
Solution StrongWolfe::zoom(ObjFunc line_func, double y0, double g0, const Solution& sol_lo,
                           double g_lo, const Solution& sol_hi, double min_step) const
    noexcept
{
    const double g_hi = line_grad(line_func, sol_hi, min_step);
    return zoom(line_func, y0, g0, sol_lo, g_lo, sol_hi, g_hi, min_step);
}

double StrongWolfe::line_grad(ObjFunc line_f, const Solution& sol, double epsi) const noexcept
{
    assert(sol.solution().size() == 1);
    double step  = sol.solution()[0];
    Solution shi = line_f({step + epsi});
    Solution slo = line_f({step - epsi});
    return (shi.fom() - slo.fom()) / (2 * epsi);
    // return (shi.fom() - sol.fom()) / epsi;
}
