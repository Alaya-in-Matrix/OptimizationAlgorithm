#include "LineSearch.h"
#include "util.h"
#include <cassert>
using namespace std;
using namespace Eigen;
LineSearch::LineSearch(ObjFunc f, ofstream& l) noexcept : _func(f), _log(l) {}
Solution ExactGoldenSelectionLineSearch::search(const Solution& sol, const VectorXd& direction,
                                                double min_walk, double max_walk) const noexcept
{
    const double min_step = min_walk / direction.lpNorm<2>();
    const double max_step = max_walk / direction.lpNorm<2>();
    assert(sol.solution().size() == static_cast<size_t>(direction.size()));
    assert(max_step > min_step);
    ObjFunc line_func = [&](const vector<double> step) -> Solution {
        Paras p = sol.solution();
        const double factor = step[0];
        for (size_t i = 0; i < p.size(); ++i) p[i] += factor * direction[i];
        return _func(p);
    };
    return Extrapolation(line_func, {0}, min_step, max_step).optimize();
}
