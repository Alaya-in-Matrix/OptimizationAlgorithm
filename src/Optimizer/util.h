#pragma once
#include <vector>
#include "Eigen/Dense"
std::vector<double> operator-(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;
std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;
std::vector<double> operator+(const std::vector<double>& v1, const Eigen::VectorXd& v2) noexcept;
std::vector<double> operator*(double, const std::vector<double>& vec) noexcept;
double vec_norm(const std::vector<double>& vec) noexcept;
double vec_norm_inf(const std::vector<double>& vec) noexcept;
