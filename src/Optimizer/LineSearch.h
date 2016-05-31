#pragma once
#include "optimizer_1d.h"
#include "obj.h"
#include "Eigen/Dense"
#include <utility>
#include <fstream>
class LineSearch
{
protected:
    typedef Eigen::VectorXd EVec;
    ObjFunc _func;
    std::ofstream& _log;

public:
    LineSearch(ObjFunc f, std::ofstream& log) noexcept;
    // min_walk 不是min_step, 而是min_step * direction.lpNorm<2>()
    virtual Solution search(const Solution& sol, const EVec& direction, double min_walk,
                            double max_walk) const noexcept = 0;
    virtual ~LineSearch(){}
};

class ExactGoldenSelectionLineSearch : public LineSearch
{
    typedef Eigen::VectorXd EVec;
public:
    using LineSearch::LineSearch;
    Solution search(const Solution& sol, const EVec& direction, double min_walk,
                    double max_walk) const noexcept;
};
