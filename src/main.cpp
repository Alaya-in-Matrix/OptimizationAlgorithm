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
    system("clear && rm -rf log");

    vector<pair<double, double>> rg_rosenbrock{{-1.5, 2}, {-0.5, 3}};
    const double inf_num = 1e6;
    puts("Rosenbrock");
    compare(rosenbrock, rg_rosenbrock);

    vector<pair<double, double>> rg_sphere(3, {-10, 10});
    puts("Sphere");
    compare(sphere, rg_sphere);

    vector<pair<double, double>> rg_beale(2, {-4.5, 4.5});
    puts("Beale");
    compare(beale, rg_beale);

    vector<pair<double, double>> rg_booth(2, {-10, 10});
    puts("Booth");
    compare(booth, rg_beale);

    vector<pair<double, double>> rg_McCormick{{-1.5, 4}, {-3, 4}};
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
template<class Algorithm>
void run_algo(ObjFunc f, const vector<pair<double, double>>& range, const Paras init, string algo_name) noexcept
{
    const double grad_epsilon = 1e-6;
    Algorithm algo(f, range, init, grad_epsilon);
    algo.clear_counter();
    Solution sol = algo.optimize();
    
    printf("fom of %s: %g, iter: %zu\n", algo_name.c_str(), sol.fom(), algo.counter());
    if(! sol.err_msg().empty())
        cout << sol.err_msg() << endl;
}
void compare(ObjFunc f, const vector<pair<double, double>>& range) noexcept
{
    Paras  init = rand_vec(range);
    Eigen::Map<Eigen::MatrixXd> mf(&init[0], 3, 1);
    cout << "init: " << Eigen::Map<Eigen::MatrixXd>(&init[0], 1, init.size()) << endl;

    run_algo<GradientDescent>(f, range, init, "GradientDescent");
    run_algo<ConjugateGradient>(f, range, init, "ConjugateGradient");
    run_algo<Newton>(f, range, init, "Newton");

    printf("===============================================\n");
}
