#include "def.h"
#include "optimizer.h"
#include "benchmark.h"
#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <cassert>
using namespace std;
mt19937_64 main_engine(RAND_SEED);
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

    vector<pair<double, double>> rg_rosenbrock{{-1.5, 2}, {-0.5, 3}};
    vector<pair<double, double>> rg_sphere(3, {-10, 10});
    vector<pair<double, double>> rg_beale(2, {-4.5, 4.5});
    vector<pair<double, double>> rg_booth(2, {-10, 10});
    vector<pair<double, double>> rg_McCormick{{-1.5, 4}, {-3, 4}};
    puts("Rosenbrock");
    compare(rosenbrock, rg_rosenbrock);
    puts("Sphere");
    compare(sphere, rg_sphere);
    puts("Beale");
    compare(beale, rg_beale);
    puts("Booth");
    compare(booth, rg_beale);
    puts("McCormick");
    compare(McCormick, rg_McCormick);


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
    if(! sol_gd.err_msg().empty())
        cout << sol_gd.err_msg() << endl;
    if(! sol_cg.err_msg().empty())
        cout << sol_cg.err_msg() << endl;
    printf("===============================================\n");
}
