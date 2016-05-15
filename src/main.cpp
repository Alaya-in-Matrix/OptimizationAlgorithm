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

    return EXIT_SUCCESS;
}
