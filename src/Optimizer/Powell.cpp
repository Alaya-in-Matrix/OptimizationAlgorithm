#include "Powell.h"
using namespace std;
using namespace Eigen;

Powell::Powell(ObjFunc f, size_t d, const Paras& i, size_t max_iter, double min_walk,
               double max_walk, string fname) noexcept
    : MultiDimOptimizer(f, d, max_iter, min_walk, max_walk, fname, "Powell"),
      _init(i)
{
}
void Powell::write_log(const Solution& s) noexcept
{
    Paras p = s.solution();
    const double y = s.fom();
    _log << endl;
    _log << "point: " << Map<MatrixXd>(&p[0], 1, _dim) << endl;
    _log << "fom:   " << y << endl;
}
Solution Powell::optimize() noexcept
{
    clear_counter();
    Solution sol = run_func(_init);
    double walk_len = numeric_limits<double>::infinity();

    // initial search directions are axes
    vector<VectorXd> search_direction(_dim, VectorXd(_dim));
    for (size_t i = 0; i < _dim; ++i)
    {
        for (size_t j = 0; j < _dim; ++j) search_direction[i][j] = i == j ? 1.0 : 0.0;
    }
    while (eval_counter() < _max_iter && walk_len > _min_walk)
    {
        double max_delta_y = -1 * numeric_limits<double>::infinity();
        size_t max_delta_id;
        Paras backup_point = sol.solution();
        for (size_t i = 0; i < _dim; ++i)
        {
            LOG(sol);
            Solution search_sol = run_line_search(sol, search_direction[i]);
            if (sol.fom() - search_sol.fom() > max_delta_y)
            {
                max_delta_y  = sol.fom() - search_sol.fom();
                max_delta_id = i;
            }
            sol = search_sol;
        }
        Paras    new_direc     = sol.solution() - backup_point;
        VectorXd new_direc_vxd = Map<VectorXd>(&new_direc[0], _dim);
        walk_len               = new_direc_vxd.lpNorm<2>();
        search_direction[max_delta_id] = new_direc_vxd;
    }
    _log << endl << "==========================================" << endl;
    write_log(sol);
    return sol;
}
Solution Powell::run_line_search(const Solution& s, const Eigen::VectorXd& direction) noexcept 
{
    Solution sol = MultiDimOptimizer::run_line_search(s, direction);
    if(sol.fom() >= s.fom())
        sol = MultiDimOptimizer::run_line_search(s, -1 * direction);
    return sol;
}
