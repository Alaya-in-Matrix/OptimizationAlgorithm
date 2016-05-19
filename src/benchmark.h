#pragma once
#include "obj.h"

Solution linear(const std::vector<double>& inp) noexcept;
Solution sphere(const std::vector<double>& inp) noexcept;
Solution rosenbrock(const std::vector<double>& inp) noexcept;
Solution beale(const Paras& inp) noexcept;
Solution booth(const Paras& inp) noexcept;
Solution McCormick(const Paras& inp) noexcept;
Solution GoldsteinPrice(const Paras& inp) noexcept;
