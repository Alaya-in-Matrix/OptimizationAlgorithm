#include "MultiDimOptimizer.h"
class NelderMead : public MultiDimOptimizer
{
    const double _alpha;
    const double _gamma;
    const double _rho;
    const double _sigma;
    const double _min_walk;
    const std::vector<Paras> _inits;
    std::vector<Solution>    _sols;
    double update_sols(size_t i, const Solution& sol) noexcept;
    void write_log(const Solution& s) noexcept;

public:
    NelderMead(ObjFunc f, size_t d, std::vector<Paras> i, double a, double g, double r, double s,
               double min_walk, size_t max_iter, std::string fname) noexcept;
    Solution optimize() noexcept;
    ~NelderMead(){}
};
