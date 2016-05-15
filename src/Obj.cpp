#include "Obj.h"
using namespace std;

Solution::Solution(const vector<double>& s, const vector<double>& cv, double fom) noexcept
    : _solution(s),
      _violation(cv),
      _fom(fom),
      _err_str("")
{}
Solution::Solution(const vector<double>& s) noexcept
    : _solution(s),
      _violation({numeric_limits<double>::infinity()}),
      _fom(numeric_limits<double>::infinity()),
      _err_str("")
{}
double Solution::fom() const noexcept { return _fom; }
double Solution::sum_violation() const noexcept
{
    double sum = 0;
    for (double vo : _violation)
    {
        sum += (vo < 0 ? 0 : vo);
    }
    return sum;
}
const Paras& Solution::solution() const noexcept { return _solution; }
const vector<double>& Solution::violations() const noexcept { return _violation; }
Solution::Solution(const Paras& s, string msg) noexcept
    : _solution(s),
      _violation({numeric_limits<double>::infinity()}),
      _fom(numeric_limits<double>::infinity()),
      _err_str(msg)
{}
bool Solution::has_error() const noexcept { return _err_str.empty(); }
string Solution::err_msg() const noexcept { return _err_str; }

