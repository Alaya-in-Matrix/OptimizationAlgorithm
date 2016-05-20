#include "def.h"
#include "optimizer.h"
#include "benchmark.h"
#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <cassert>
#include <string>
using namespace std;
mt19937_64 main_engine(RAND_SEED);
Paras rand_vec(const vector<pair<double, double>>& rg) noexcept;
void compare(ObjFunc f, const vector<pair<double, double>>& range, string) noexcept;
int main()
{
    // vector<pair<double, double>> rg{{-1, 3}};
    // FibOptimizer fbo(sphere, rg);
    // GoldenSelection gso(sphere, rg);
    // Extrapolation exo(sphere, rg);
    // cout << "result of fib: " << fbo.optimize().solution().front() << endl;
    // cout << "result of gso: " << gso.optimize().solution().front() << endl;
    // cout << "result of exo: " << exo.optimize().solution().front() << endl;
    system("clear && rm -rf *.log");

    vector<pair<double, double>> rg_rosenbrock{{-1.5, 2}, {-0.5, 3}};
    compare(rosenbrock, rg_rosenbrock, "Rosenbrock");

    vector<pair<double, double>> rg_sphere(3, {-10, 10});
    compare(sphere, rg_sphere, "Sphere");

    vector<pair<double, double>> rg_beale(2, {-4.5, 4.5});
    compare(beale, rg_beale, "Beale");

    vector<pair<double, double>> rg_booth(2, {-10, 10});
    compare(booth, rg_beale, "Booth");

    vector<pair<double, double>> rg_McCormick{{-1.5, 4}, {-3, 4}};
    compare(McCormick, rg_McCormick, "McCormick");

    vector<pair<double, double>> rg_GoldsteinPrice{{-2, 2}, {-2, 2}};
    compare(GoldsteinPrice, rg_GoldsteinPrice, "GoldsteinPrice");


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
template<class Algorithm>
void run_algo(ObjFunc f, const vector<pair<double, double>>& range, const Paras init, string algo_name, string fname) noexcept
{
    const double grad_epsilon = 1e-5;
    const double zero_grad    = 5e-2;
    const size_t max_iter     = 1000;
    Algorithm algo(f, range, init, grad_epsilon, zero_grad, max_iter, fname, algo_name);
    Solution sol = algo.optimize();
    
    printf("fom of %s: %g, iter: %zu\n", algo_name.c_str(), sol.fom(), algo.counter());
}
void compare(ObjFunc f, const vector<pair<double, double>>& range, string fname) noexcept
{
    puts(fname.c_str());
    Paras  init = rand_vec(range);
    Eigen::Map<Eigen::MatrixXd> mf(&init[0], 3, 1);
    cout << "init: " << Eigen::Map<Eigen::MatrixXd>(&init[0], 1, init.size()) << endl;

    run_algo<GradientDescent>(f, range, init, "GradientDescent", fname);
    run_algo<ConjugateGradient>(f, range, init, "ConjugateGradient", fname);
    run_algo<Newton>(f, range, init, "Newton", fname);

    printf("===============================================\n");
}
