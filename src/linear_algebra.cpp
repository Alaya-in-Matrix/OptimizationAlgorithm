#include "linear_algebra.h"
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;
vector<double> operator-(const vector<double>& v1, const vector<double>& v2) noexcept
{
    if (v1.size() != v2.size())
    {
        cerr << "v1.size() != v2.size() in 'v1 + v2'" << endl;
        exit(EXIT_FAILURE);
    }
    vector<double> v = v1;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        v[i] -= v2[i];
    }
    return v;
}
vector<double> operator+(const vector<double>& v1, const vector<double>& v2) noexcept
{
    if (v1.size() != v2.size())
    {
        cerr << "v1.size() != v2.size() in 'v1 + v2'" << endl;
        exit(EXIT_FAILURE);
    }
    vector<double> v = v1;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        v[i] += v2[i];
    }
    return v;
}
vector<double> operator*(const double factor, const vector<double>& vec) noexcept
{
    vector<double> v = vec;
    for (auto& val : v) val *= factor;
    return v;
}
double vec_norm(const vector<double>& vec) noexcept
{
    double sum_square = 0;
    for (auto v : vec) sum_square += v * v;
    return sqrt(sum_square);
}
double vec_norm_inf(const vector<double>& vec) noexcept
{
    return fabs(*max_element(vec.begin(), vec.end()));
}
