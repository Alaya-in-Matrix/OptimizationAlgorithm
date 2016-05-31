#include "MultiDimOptimizer.h"
class NelderMead : public MultiDimOptimizer
{
    const double _alpha;
    const double _gamma;
    const double _rho;
    const double _sigma;
    const double _converge_len;
    const std::vector<Paras> _inits;
    std::vector<Solution>    _sols;
    double max_simplex_len() const noexcept;
    void write_log(const Solution& s) noexcept;

public:
    NelderMead(ObjFunc f, size_t d, std::vector<Paras> i, double a, double g, double r, double s,
               double conv_len, size_t max_iter, std::string fname) noexcept;
    Solution optimize() noexcept;
    ~NelderMead(){}
};
