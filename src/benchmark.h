#pragma once
#include "Optimizer/obj.h"

Solution linear(const std::vector<double>& inp) noexcept;
Solution sphere(const std::vector<double>& inp) noexcept;
Solution rosenbrock(const std::vector<double>& inp) noexcept;
Solution beale(const Paras& inp) noexcept;
Solution booth(const Paras& inp) noexcept;
Solution GoldsteinPrice(const Paras& inp) noexcept;
Solution ellip(const Paras& inp) noexcept;
Solution matyas(const Paras& inp) noexcept;
Solution threeHumpCamel(const Paras& inp) noexcept;
Solution Himmelblau(const Paras& inp) noexcept;
