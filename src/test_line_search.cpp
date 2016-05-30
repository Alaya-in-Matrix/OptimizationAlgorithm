#include "line_search.h"
#include "benchmark.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
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
Solution cubic(const Paras& inp) noexcept
{
    double x = inp[0];
    double fom = pow(x-1, 3) + 5 * pow(x-1, 2) + x;
    return Solution(inp, {0}, fom);
}
void test(ObjFunc, ofstream& l, string fname, double p0);
int main()
{
    ofstream log("log");
    test(Cos,   log, "cos", 0.1);
    test(quad1, log, "quad1", 0.1);
    test(quad2, log, "quad2", 0.1);

    Solution sp = sphere({2, 3});
    VectorXd direction(2);
    direction << -2, -3;
    StrongWolfe swo(sphere, log, 1e-4, 0.9);
    Solution ssw = swo.search(sp, direction, 1e-5, 10);
    cout << "sphere" << endl;
    cout << ssw.fom() << endl;
    log.close();

    return EXIT_SUCCESS;
}

void test(ObjFunc f, ofstream& l, string fname, double p0)
{
    cout << fname << endl;
    Solution sp = f({p0});
    VectorXd direction(1);
    direction << 1;
    // ExactGoldenSelectionLineSearch egs(f, l);
    // Solution seg = egs.search(sp, direction, 1e-5, 2*3.14159);
    // cout << seg.solution()[0] << ", " << seg.fom() << endl;

    StrongWolfe swo(f, l, 1e-4, 0.9);
    Solution ssw = swo.search(sp, direction, 1e-5, 2*3.14159);
    cout << ssw.solution()[0] << ", " << ssw.fom() << endl;
}

