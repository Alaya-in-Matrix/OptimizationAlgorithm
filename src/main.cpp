#include "optimizer.h"
#include "benchmark.h"
#include <iostream>
#include <utility>
#include <vector>
#include <random>
using namespace std;
mt19937_64 main_engine(random_device{}());
Paras rand_vec(const vector<pair<double, double>>& rg) noexcept;
void compare(ObjFunc f, const vector<pair<double, double>>& range) noexcept;
int main()
{
    // vector<pair<double, double>> rg{{-1, 3}};
    // FibOptimizer fbo(sphere, rg);
    // GoldenSelection gso(sphere, rg);
    // Extrapolation exo(sphere, rg);
    // cout << "result of fib: " << fbo.optimize().solution().front() << endl;
    // cout << "result of gso: " << gso.optimize().solution().front() << endl;
    // cout << "result of exo: " << exo.optimize().solution().front() << endl;

    vector<pair<double, double>> rg_rosenbrock(2, {-10, 10});
    vector<pair<double, double>> rg_sphere(3, {-10, 10});
    puts("Rosenbrock");
    compare(rosenbrock, rg_rosenbrock);
    puts("Sphere");
    compare(sphere, rg_sphere);


    return EXIT_SUCCESS;
}
Paras rand_vec(const vector<pair<double, double>>& rg) noexcept
{
    Paras rand_init(rg.size(), 0);
    for(size_t i = 0; i < rg.size(); ++i)
    {
        const double lb = rg[i].first;
        const double ub = rg[i].second;
        rand_init[i]    = uniform_real_distribution<double>(lb, ub)(main_engine);
    }
    return rand_init;
}
void compare(ObjFunc f, const vector<pair<double, double>>& range) noexcept
{
    const Paras  init         = rand_vec(range);
    const double grad_epsilon = 1e-6;

    GradientDescent   gdo(f, range, init, grad_epsilon);
    ConjugateGradient cgo(f, range, init, grad_epsilon);

    Solution sol_gd = gdo.optimize();
    Solution sol_cg = cgo.optimize();

    printf("fom of GradientDescent: %g, iter %zu\n", sol_gd.fom(), gdo.counter());
    printf("fom of ConjugateGradient: %g, iter %zu\n", sol_cg.fom(), cgo.counter());
    printf("===============================================\n");
}
