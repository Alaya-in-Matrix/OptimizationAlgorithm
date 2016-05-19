#include "benchmark.h"
#include <functional>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
using namespace std;

Solution linear(const vector<double>& inp) noexcept
{
    if(inp.empty())
        return Solution(inp, "Empty input for linear function");

    double y = 0;
    for(auto x : inp)
        y += x;
    return Solution(inp, {0}, y);
}
Solution sphere(const vector<double>& inp) noexcept
{
    if(inp.empty())
        return Solution(inp, "Empty input for sphere function");

    double y = 0;
    for(auto x : inp)
        y += pow(x, 2);
    return Solution(inp, {0}, y);
}
Solution rosenbrock(const vector<double>& inp) noexcept
{
    if(inp.size() != 2)
    {
        return Solution(inp, "Rosenbrock function is 2-D function, while the input size is " +
                                 to_string(inp.size()));
    }
    
    const double x = inp[0];
    const double y = inp[1];
    
    // (1-x)^2 + 100 * (y - x^2)^2
    const double fom = pow(1-x, 2) + 100 * pow((y - pow(x, 2)), 2);
    return Solution(inp, {0}, fom);
}
Solution beale(const Paras& inp) noexcept
{
    if(inp.size() != 2)
    {
        return Solution(inp, "Beale function is 2-D function, while the input size is " +
                                 to_string(inp.size()));
    }

    const double x   = inp[0];
    const double y   = inp[1];
    const double fom = pow((1.5 - x + x * y), 2) + pow((2.25 - x + x * pow(y, 2)), 2) + pow((2.625 - x + x * pow(y, 3)), 2);
    return Solution(inp, {0}, fom);
}
Solution booth(const Paras& inp) noexcept
{
    if(inp.size() != 2)
    {
        return Solution(inp, "Booth function is 2-D function, while the input size is " +
                                 to_string(inp.size()));
    }

    const double x   = inp[0];
    const double y   = inp[1];
    const double fom = pow(x - 2*y -7, 2) + pow(2*x + y - 5, 2);
    return Solution(inp, {0}, fom);
}
Solution McCormick(const Paras& inp) noexcept
{
    if(inp.size() != 2)
    {
        return Solution(inp, "McCormick function is 2-D function, while the input size is " +
                                 to_string(inp.size()));
    }
    // global: (-0.54719, 1.54719) => -1.9133
    const double x   = inp[0];
    const double y   = inp[1];
    const double fom = sin(x+y) + pow(x-y, 2) - 1.5*x + 2.5*y + 1;
    return Solution(inp, {0}, fom);
}
Solution GoldsteinPrice(const Paras& inp) noexcept
{
    if(inp.size() != 2)
    {
        return Solution(inp, "GoldsteinPrice function is 2-D function, while the input size is " +
                                 to_string(inp.size()));
    }
    // global: (0, -1) => 0
    const double x   = inp[0];
    const double y   = inp[1];
    const double fom = -3 + (1+pow(x+y+1, 2)*(19-14*x+3*pow(x, 2)-14*y+6*x*y+3*pow(y, 2))) * (30 + pow(2*x-3*y, 2)*(18-32*x+12*pow(x, 2)+48*y-36*x*y+27*pow(y, 2)));
    return Solution(inp, {0}, fom);
}
