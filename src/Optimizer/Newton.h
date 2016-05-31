#pragma once
#include "MultiDimOptimizer.h"
class Newton : public GradientMethod
{
    void write_log(const Solution& s, const Eigen::VectorXd& grad,
                   const Eigen::MatrixXd& hess) noexcept;

public:
    TYPICAL_DEF(Newton);
};
