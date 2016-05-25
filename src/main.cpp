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
size_t rseed = RAND_SEED;
mt19937_64 main_engine(rseed);
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
    cout << "seed: " << rseed << endl;

    vector<pair<double, double>> rg_rosenbrock{{-1.5, 2}, {-0.5, 3}};
    compare(rosenbrock, rg_rosenbrock, "Rosenbrock");

    // vector<pair<double, double>> rg_sphere(3, {-10, 10});
    // compare(sphere, rg_sphere, "Sphere");

    vector<pair<double, double>> rg_beale{{0, 4.5}, {-4.5, 4.5}};
    compare(beale, rg_beale, "Beale");

    // vector<pair<double, double>> rg_booth(2, {-10, 10});
    // compare(booth, rg_beale, "Booth");

    vector<pair<double, double>> rg_GoldsteinPrice{{-2, 2}, {-2, 2}};
    compare(GoldsteinPrice, rg_GoldsteinPrice, "GoldsteinPrice");

    vector<pair<double, double>> rg_ellip(10, {-10, 10});
    compare(ellip, rg_ellip, "Ellip");

    // vector<pair<double, double>> rg_matyas(2, {-10, 10});
    // compare(matyas, rg_matyas, "Matyas");

    // vector<pair<double, double>> rg_camel(2, {-5, 5});
    // compare(threeHumpCamel, rg_camel, "ThreeHumpCamel");

    vector<pair<double, double>> rg_himmel(2, {-5, 5});
    compare(Himmelblau, rg_himmel, "Himmelblau");

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
void run_grad_algo(ObjFunc f, const vector<pair<double, double>>& range, const Paras init, string algo_name, string fname) noexcept
{
    const double grad_epsilon = 1e-5;
    const double zero_grad    = 1e-2;
    const double min_walk     = 5e-6;
    const double max_walk     = 50;
    const size_t max_iter     = 1000;
    const size_t dim          = range.size();
    Algorithm algo(f, dim, init, grad_epsilon, zero_grad, min_walk, max_walk, max_iter, fname);
    Solution sol = algo.optimize();
    
    printf("fom of %s: %g, iter: %zu\n", algo_name.c_str(), sol.fom(), algo.counter());
}
void run_simplex(ObjFunc f, const vector<pair<double, double>>& range, string fname) noexcept
{
    const double alpha    = 1;
    const double gamma    = 2;
    const double rho      = 0.5;
    const double sigma    = 0.5;
    const double conv_len = 1e-3;
    const size_t max_iter = 200;
    vector<Paras> inits(range.size() + 1);
    for(auto& iv : inits)
        iv = rand_vec(range);
    NelderMead nmo(f, range.size(), inits, alpha, gamma, rho, sigma, conv_len, max_iter, fname);
    Solution sol = nmo.optimize();
    printf("fom of NelderMead: %g, iter: %zu\n", sol.fom(), nmo.counter());
}
void compare(ObjFunc f, const vector<pair<double, double>>& range, string fname) noexcept
{
    puts(fname.c_str());
    Paras  init = rand_vec(range);
    Eigen::Map<Eigen::MatrixXd> mf(&init[0], 3, 1);
    cout << "init: " << Eigen::Map<Eigen::MatrixXd>(&init[0], 1, init.size()) << endl;

    run_grad_algo<GradientDescent>(f, range, init, "GradientDescent", fname);
    run_grad_algo<ConjugateGradient>(f, range, init, "ConjugateGradient", fname);
    run_grad_algo<Newton>(f, range, init, "Newton", fname);
    run_grad_algo<DFP>(f, range, init, "DFP", fname);
    run_grad_algo<BFGS>(f, range, init, "BFGS", fname);
    run_simplex(f, range, fname);

    printf("===============================================\n");
}
