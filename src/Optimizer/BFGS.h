#include "MultiDimOptimizer.h"
class BFGS : public GradientMethod
{
    void write_log(const Solution&, const Eigen::VectorXd& grad,
                   const Eigen::MatrixXd& quasi_hess) noexcept;

public:
    TYPICAL_DEF(BFGS);
};
