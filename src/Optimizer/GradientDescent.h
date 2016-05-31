#pragma once
#include "MultiDimOptimizer.h"
class GradientDescent : public GradientMethod
{
    void write_log(const Solution&, const Eigen::VectorXd& grad) noexcept;
public:
    TYPICAL_DEF(GradientDescent);
};
