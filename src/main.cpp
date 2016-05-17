#include "optimizer.h"
#include "benchmark.h"
#include <iostream>
#include <utility>
#include <vector>
#include <random>
using namespace std;
mt19937_64 main_engine(random_device{}());

int main()
{
    // vector<pair<double, double>> rg{{-1, 3}};
    // FibOptimizer fbo(sphere, rg);
    // GoldenSelection gso(sphere, rg);
    // Extrapolation exo(sphere, rg);
    // cout << "result of fib: " << fbo.optimize().solution().front() << endl;
    // cout << "result of gso: " << gso.optimize().solution().front() << endl;
    // cout << "result of exo: " << exo.optimize().solution().front() << endl;

    vector<pair<double, double>> rgRosenbrock{{-10, 10}, {-10, 10}};
    Paras rand_init(rgRosenbrock.size(), 0);
    for(size_t i = 0; i < rgRosenbrock.size(); ++i)
    {
        const double lb = rgRosenbrock[i].first;
        const double ub = rgRosenbrock[i].second;
        rand_init[i]    = uniform_real_distribution<double>(lb, ub)(main_engine);
    }
    GradientDescent gdo(rosenbrock, rgRosenbrock, rand_init, 1e-6);
    Solution solGradientDescend = gdo.optimize();
    cout << "result of gdo: " ;
    for(double v : solGradientDescend.solution())
    {
        cout << v << ' ';
    }
    cout << endl;

    ConjugateGradient cgo(rosenbrock, rgRosenbrock, rand_init, 1e-6);
    Solution solConjGrad = cgo.optimize();
    cout << "result of cgo: " ;
    for(double v : solConjGrad.solution())
    {
        cout << v << ' ';
    }
    cout << endl;

    return EXIT_SUCCESS;
}
