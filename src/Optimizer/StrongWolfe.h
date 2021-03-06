#pragma once
#include "LineSearch.h"
class StrongWolfe : public LineSearch
{
    const double _c1;
    const double _c2;

    Solution zoom(ObjFunc line_f, double y0, double g0, const Solution& sol_lo, double g_lo,
                  const Solution& sol_hi, double g_hi, double min_step) const noexcept;
    Solution zoom(ObjFunc line_f, double y0, double g0, const Solution& sol_lo, double g_lo,
                  const Solution& sol_hi, double min_step) const noexcept;
    EVec cubic_interpolation(double x1, double y1, double g1, double x2, double y2, double g2, bool cubic = true) const noexcept;
    double cubic_predict(double x1, double y1, double g1, double x2, double y2, double g2, bool cubic = true) const noexcept;
    double line_grad(ObjFunc line_f, const Solution& sol, double epsi) const noexcept;

public:
    StrongWolfe(ObjFunc f, std::ofstream& log, double c1, double c2) noexcept;
    Solution search(const Solution& sol, const EVec& direction, double min_walk,
                    double max_walk) const noexcept;
};
