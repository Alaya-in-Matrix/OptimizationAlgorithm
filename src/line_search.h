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
    virtual Solution search(const Solution& sol, const EVec& direction, double min_walk, double max_walk) = 0;
};

class ExactGoldenSelectionLineSearch : public LineSearch
{
    typedef Eigen::VectorXd EVec;
public:
    using LineSearch::LineSearch;
    Solution search(const Solution& sol, const EVec& direction, double min_walk, double max_walk);
};
