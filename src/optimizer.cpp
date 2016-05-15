#include "optimizer.h"
#include <random>
#include <cstdio>
#include <iostream>
using namespace std;

mt19937_64 engine(random_device{}());

Solution FibOptimizer::optimize() const noexcept
{
    // 1-D function
    // function shoulde be convex function
    if(_ranges.size() != 1)
    {
        return Solution({}, "FibOptimizer requires 1D function, while the actual dim is " + to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if(a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }
    
    const size_t iter = 16;
    vector<double> fib_list{1, 1};
    if(iter > 2)
    {
        fib_list.reserve(iter);
        for(size_t i = 2; i < iter; ++i)
            fib_list.push_back(fib_list[i-1] + fib_list[i-2]);
    }

    for(size_t i = iter - 1; i >0; --i)
    {
        const double rate = fib_list[i - 1] / fib_list[i];
        cout << rate << endl;

        const double y1 = _func({a1}).fom();
        const double y2 = _func({a2}).fom();

        if(y1 < y2)
            a2 = a1 + rate * (a2 - a1);
        else
            a1 = a2 + rate * (a1 - a2);
        // printf("[%g, %g]\n", a1, a2);
    }

    return _func({a1});
}
Solution GoldenSelection::optimize() const noexcept
{
    // 1-D function
    // function shoulde be convex function
    if(_ranges.size() != 1)
    {
        return Solution({}, "GoldenSelection requires 1D function, while the actual dim is " + to_string(_ranges.size()));
    }
    double a1 = _ranges.front().first;
    double a2 = _ranges.front().second;
    if(a1 > a2)
    {
        return Solution({}, "Range is [" + to_string(a1) + ", " + to_string(a2) + "]");
    }
    
    const size_t iter = 16;
    const double rate = (sqrt(5) - 1) / 2;

    for(size_t i = iter - 1; i >0; --i)
    {

        const double y1 = _func({a1}).fom();
        const double y2 = _func({a2}).fom();

        if(y1 < y2)
            a2 = a1 + rate * (a2 - a1);
        else
            a1 = a2 + rate * (a1 - a2);
        // printf("[%g, %g]\n", a1, a2);
    }

    return _func({a1});
}
