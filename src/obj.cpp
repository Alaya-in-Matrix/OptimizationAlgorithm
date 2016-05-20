#include "obj.h"
#include <limits>
using namespace std;

Solution::Solution(const vector<double>& s, const vector<double>& cv, double fom) noexcept
    : _solution(s),
      _violation(cv),
      _fom(fom)
{}
Solution::Solution(const vector<double>& s) noexcept
    : _solution(s),
      _violation({numeric_limits<double>::infinity()}),
      _fom(numeric_limits<double>::infinity())
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
