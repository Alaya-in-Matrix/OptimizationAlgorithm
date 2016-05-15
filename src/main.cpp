#include "optimizer.h"
#include "benchmark.h"
#include <iostream>
#include <utility>
#include <vector>
using namespace std;

int main()
{
    vector<pair<double, double>> rg{{-1, 3}};
    FibOptimizer fbo(linear, rg);

    Solution result = fbo.optimize();

    cout << result.solution().front() << endl;

    return EXIT_SUCCESS;
}
