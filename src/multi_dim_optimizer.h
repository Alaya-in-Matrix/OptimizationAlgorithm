#pragma once
#include "optimizer_1d.h"
class MultiDimOptimizer
{
private:
    ObjFunc _func;

protected:
    const size_t _dim;
    const size_t _max_iter;
    const double _min_walk;
    const double _max_walk;
    const std::string _func_name;
    const std::string _algo_name;
    size_t _eval_counter;
    size_t _linesearch_counter;
    std::ofstream _log;
    virtual Solution run_func(const Paras&) noexcept;
    virtual Solution line_search(const Paras& point, const Eigen::VectorXd& direc) noexcept;
    virtual Solution armijo_bracketing_linesearch(
        const Paras& point, const Eigen::VectorXd& direc,
        double guess = std::numeric_limits<double>::infinity()) noexcept;
    // virtual Solution line_search(const Paras& point, const Eigen::VectorXd& direc,
    //                              const Eigen::VectorXd& grad) noexcept;

public:
    void clear_counter() noexcept { _eval_counter = 0; _linesearch_counter = 0;}
    size_t eval_counter() noexcept { return _eval_counter; }
    size_t linesearch_counter() noexcept { return _linesearch_counter; }
    MultiDimOptimizer(ObjFunc f, size_t d, size_t max_iter, double min_walk, double max_walk,
                      std::string func_name, std::string algo_name) noexcept;
    virtual ~MultiDimOptimizer(){}
};
class GradientMethod : public MultiDimOptimizer
{
protected:
    const Paras         _init;
    const double        _epsilon; // use _epsilon to calc gradient
    const double        _zero_grad; // threshold to judge whether gradient is zero

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
class GradientDescent : public GradientMethod
{
    void write_log(const Solution&, const Eigen::VectorXd& grad) noexcept;
public:
    TYPICAL_DEF(GradientDescent);
};
class ConjugateGradient : public GradientMethod
{
    void write_log(const Solution&, const Eigen::VectorXd& grad,
                   const Eigen::VectorXd& conj_grad) noexcept;

public:
    TYPICAL_DEF(ConjugateGradient);
};
class Newton : public GradientMethod
{
    void write_log(const Solution& s, const Eigen::VectorXd& grad,
                   const Eigen::MatrixXd& hess) noexcept;

public:
    TYPICAL_DEF(Newton);
};
class DFP : public GradientMethod
{
    void write_log(const Solution& s, const Eigen::VectorXd& grad,
                   const Eigen::MatrixXd& quasi_hess) noexcept;

public:
    TYPICAL_DEF(DFP);
};
class BFGS : public GradientMethod
{
    void write_log(const Solution&, const Eigen::VectorXd& grad,
                   const Eigen::MatrixXd& quasi_hess) noexcept;

public:
    TYPICAL_DEF(BFGS);
};
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
};
class Powell : public MultiDimOptimizer
{
    Paras _init;
    void write_log(const Solution& s) noexcept;

public:
    Powell(ObjFunc f, size_t d, const Paras& i, size_t max_iter, double min_walk, double max_walk,
           std::string fname) noexcept;
    Solution optimize() noexcept;
};
