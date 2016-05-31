#include "MultiDimOptimizer.h"
class DFP : public GradientMethod
{
    void write_log(const Solution& s, const Eigen::VectorXd& grad,
                   const Eigen::MatrixXd& quasi_hess) noexcept;

public:
    TYPICAL_DEF(DFP);
};
