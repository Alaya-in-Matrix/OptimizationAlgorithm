#include "line_search.h"
#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;
using namespace Eigen;
Solution Cos(const Paras& inp) noexcept
{
    // y = cos(x)
    double fom = cos(inp[0]);
    return Solution(inp, {0}, fom);
}
Solution quad1(const Paras& inp) noexcept
{
    // y = (x -1 )^2
    double fom = pow(inp[0] - 1, 2);
    return Solution(inp, {0}, fom);
}
Solution quad2(const Paras& inp) noexcept
{
    double fom = 1 - pow(inp[0], 2);
    return Solution(inp, {0}, fom);
}
void test(ObjFunc, ofstream& l, string fname);
int main()
{
    ofstream log("log");
    log.close();

    test(Cos,   log, "cos");
    test(quad1, log, "quad1");
    test(quad2, log, "quad2");

    return EXIT_SUCCESS;
}

template<typename S> void run_search(ObjFunc f, ofstream& l, double min_walk, double max_walk)
{
    S s(f, l);
    Solution sp  = f({0});
    VectorXd direction(1);
    direction << 1;
    Solution sol = s.search(sp, direction, min_walk, max_walk);

    double x = sol.solution()[0];
    double y = sol.fom();
    cout << x << ", " << y << endl;
}

void test(ObjFunc f, ofstream& l, string fname)
{
    cout << fname << endl;
    Solution sp = f({0});
    VectorXd direction(1);
    direction << 1;
    ExactGoldenSelectionLineSearch egs(f, l);
    Solution seg = egs.search(sp, direction, 1e-5, 2*3.14159);
    cout << seg.solution()[0] << ", " << seg.fom() << endl;

    StrongWolfe swo(f, l, 1e-4, 0.9);
    Solution ssw = swo.search(sp, direction, 1e-5, 2*3.14159);
    cout << ssw.solution()[0] << ", " << ssw.fom() << endl;
}

