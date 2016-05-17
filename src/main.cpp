#include "optimizer.h"
#include "benchmark.h"
#include <iostream>
#include <utility>
#include <vector>
using namespace std;

int main()
{
    vector<pair<double, double>> rg{{-1, 3}};
    FibOptimizer fbo(sphere, rg);
    GoldenSelection gso(sphere, rg);
    Extrapolation exo(sphere, rg);
    cout << "result of fib: " << fbo.optimize().solution().front() << endl;
    cout << "result of gso: " << gso.optimize().solution().front() << endl;
    cout << "result of exo: " << exo.optimize().solution().front() << endl;

    vector<pair<double, double>> rgGradientDescend{{-10, 10}, {-10, 10}};
    GradientDescent gdo(rosenbrock, rgGradientDescend, 1e-6);
    Solution solGradientDescend = gdo.optimize();
    cout << "result of gdo: " ;
    for(double v : solGradientDescend.solution())
    {
        cout << v << ' ';
    }
    cout << endl;
    GradientDescent gdo_sphere(sphere, {{-1, 3}, {-1, 3}, {-1, 3}}, 1e-4);
    Solution sol_sphere = gdo_sphere.optimize();
    cout << "result of gdo_sphere: ";
    for(auto v : sol_sphere.solution())
    {
        cout << v << ' ';
    }
    cout << endl;

    return EXIT_SUCCESS;
}
