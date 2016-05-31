#pragma once
#include "multi_dim_optimizer.h"
class Powell : public MultiDimOptimizer
{
    Paras _init;
    void  write_log(const Solution& s) noexcept;
    Solution run_line_search(const Solution& s, const Eigen::VectorXd& direction) noexcept;
    
public:
    Powell(ObjFunc f, size_t d, const Paras& i, size_t max_iter, double min_walk, double max_walk,
           std::string fname) noexcept;
    Solution optimize() noexcept;
    ~Powell(){}
};
