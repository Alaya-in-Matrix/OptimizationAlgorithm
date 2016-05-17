#pragma once
#include <vector>
std::vector<double> operator-(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;
std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;
std::vector<double> operator*(double, const std::vector<double>& vec) noexcept;
double vec_norm(const std::vector<double>& vec) noexcept;
