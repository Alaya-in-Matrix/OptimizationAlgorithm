#include "MultiDimOptimizer.h"
class ConjugateGradient : public GradientMethod
{
    void write_log(const Solution&, const Eigen::VectorXd& grad,
                   const Eigen::VectorXd& conj_grad) noexcept;

public:
    TYPICAL_DEF(ConjugateGradient);
};
