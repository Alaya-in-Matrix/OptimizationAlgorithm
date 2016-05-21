#include "benchmark.h"
#include "Eigen/Dense"
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
    {
        cerr << "Empty input for linear function" << endl;
        exit(EXIT_FAILURE); 
    }

    double y = 0;
    for(auto x : inp)
        y += x;
    return Solution(inp, {0}, y);
}
Solution sphere(const vector<double>& inp) noexcept
{
    if(inp.empty())
    {
        cerr << "Empty input for sphere function" << endl;
        exit(EXIT_FAILURE);
    }

    double y = 0;
    for(auto x : inp)
        y += pow(x, 2);
    return Solution(inp, {0}, y);
}
Solution rosenbrock(const vector<double>& inp) noexcept
{
    if(inp.size() != 2)
    {
        cerr << "Rosenbrock function is 2-D function" << endl;
        exit(EXIT_FAILURE);
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
        cerr << "Beale function is 2-D function" << endl;
        exit(EXIT_FAILURE);
    }

    const double x   = inp[0];
    const double y   = inp[1];
    const double fom = pow((1.5 - x + x * y), 2) + pow((2.25 - x + x * pow(y, 2)), 2) +
                       pow((2.625 - x + x * pow(y, 3)), 2);
    return Solution(inp, {0}, fom);
}
Solution booth(const Paras& inp) noexcept
{
    if(inp.size() != 2)
    {
        cerr << "Booth function is 2-D function" << endl;
        exit(EXIT_FAILURE);
    }

    const double x   = inp[0];
    const double y   = inp[1];
    const double fom = pow(x - 2*y -7, 2) + pow(2*x + y - 5, 2);
    return Solution(inp, {0}, fom);
}
Solution GoldsteinPrice(const Paras& inp) noexcept
{
    if(inp.size() != 2)
    {
        cerr << "GoldsteinPrice function is 2-D function" << endl;
        exit(EXIT_FAILURE);
    }
    // global: (0, -1) => 0
    const double x   = inp[0];
    const double y   = inp[1];
    const double fom = -3 
        + (1  + pow(x + y + 1, 2) * (19 - 14 * x + 3 * pow(x, 2) - 14 * y + 6 * x * y + 3 * pow(y, 2))) 
        * (30 + pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * pow(x, 2) + 48 * y - 36 * x * y + 27 * pow(y, 2)));
    return Solution(inp, {0}, fom);
}

using namespace Eigen;
Solution ellip(const Paras& inp) noexcept
{
    if(inp.size() != 2)
    {
        cerr << "Ellip function is 2-D function" << endl;
        exit(EXIT_FAILURE);
    }
    Paras p = inp;
    VectorXd vec = Map<VectorXd>(&p[0], 2, 1);
    MatrixXd mat(2, 2);
    mat(0,0) = sqrt(2);
    mat(0,1) = -1*sqrt(2);
    mat(1,0) = sqrt(2);
    mat(1,1) = sqrt(2);

    VectorXd nvec = mat * vec;

    double x = nvec(0);
    double y = nvec(1);
    const double fom = pow(x, 2) + pow(y, 2) / 400;
    return Solution(inp, {0}, fom);
}
