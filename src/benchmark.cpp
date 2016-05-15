#include "benchmark.h"
#include <functional>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
using namespace std;

Solution sphere(const vector<double>& inp) noexcept
{
    if(inp.empty())
        return Solution(inp, "Empty input for sphere function");

    double y = 0;
    for(auto x : inp)
        y += x * x;
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
