#pragma once
#include<random>
#include <iostream>
// These two macros are controlled by CMake

#ifndef RAND_SEED
#ifdef DEBUG_OPTIMIZER
// the value of this macro is irrevalent
#define RAND_SEED 1270865179
#else
#define RAND_SEED std::random_device{}()
#endif
#endif
