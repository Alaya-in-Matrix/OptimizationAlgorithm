#pragma once
#include "optimizer_1d.h"
#include "StrongWolfe.h"
#ifdef WRITE_LOG
#define LOG(...) write_log(__VA_ARGS__)
#else
#define LOG(p)
#endif
class MultiDimOptimizer
{
protected:
    const size_t      _dim;
    const size_t      _max_iter;
    const double      _min_walk;
    const double      _max_walk;
    const std::string _func_name;
    const std::string _algo_name;
    std::ofstream     _log;

    virtual Solution run_func(const Paras&) noexcept;
    virtual Solution run_line_search(const Solution& s, const Eigen::VectorXd& direction) noexcept;

private:
    ObjFunc     _func;
    StrongWolfe _line_searcher;
    size_t      _eval_counter;
    size_t      _linesearch_counter;

public:
    void   clear_counter() noexcept { _eval_counter = 0; _linesearch_counter = 0;}
    size_t eval_counter() noexcept { return _eval_counter; }
    size_t linesearch_counter() noexcept { return _linesearch_counter; }
    MultiDimOptimizer(ObjFunc f, size_t d, size_t max_iter, double min_walk, double max_walk,
                      std::string func_name, std::string algo_name) noexcept;
    virtual ~MultiDimOptimizer(){}
};
class GradientMethod : public MultiDimOptimizer
{
protected:
    const Paras  _init;
    const double _epsilon; // use _epsilon to calc gradient
    const double _zero_grad; // threshold to judge whether gradient is zero

    // virtual Eigen::VectorXd get_gradient(const Paras& p)    noexcept;
    virtual Eigen::VectorXd get_gradient(const Solution& s) noexcept;
    // virtual Eigen::MatrixXd hessian(const Paras& point) noexcept;
    virtual Eigen::MatrixXd hessian(const Solution& point, const Eigen::VectorXd& grad) noexcept;

public:
    GradientMethod(ObjFunc f, size_t d, Paras i, double epsi, double zgrad, double minwalk,
                   double maxwalk, size_t max_iter, std::string fname, std::string aname) noexcept;
    virtual ~GradientMethod() { if(_log.is_open()) _log.close(); } 
};

#define TYPICAL_DEF(ClassName)                                                                \
    ClassName(ObjFunc f, size_t d, Paras i, double epsi, double zgrad, double minwalk,        \
              double maxwalk, size_t max_iter, std::string fname)                             \
        : GradientMethod(f, d, i, epsi, zgrad, minwalk, maxwalk, max_iter, fname, #ClassName) \
    {}                                                                                        \
    Solution optimize() noexcept;                                                             \
    ~ClassName() {}
