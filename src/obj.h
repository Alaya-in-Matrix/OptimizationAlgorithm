#pragma once
#include <vector>
#include <string>
#include <functional>

typedef std::vector<double> Paras;
class Solution
{
    // Para与evaluated result放在一个class中，方便(partial) sort
    Paras _solution;
    std::vector<double> _violation;  // sum of constraint violation
    double _fom;
    std::string _err_str;

public:
    Solution(const Paras& s, const std::vector<double>& cv, double fom) noexcept;
    explicit Solution(const Paras& s) noexcept;
    Solution() =delete;
    Solution(const Paras& s, std::string) noexcept;
    double fom() const noexcept;
    double sum_violation() const noexcept;
    const std::vector<double>& violations() const noexcept;
    const Paras& solution() const noexcept;
    Solution& operator=(const Solution&) =default;
    bool has_error() const noexcept;
    std::string err_msg() const noexcept;
};
typedef std::function<Solution(const std::vector<double>&)> ObjFunc;
