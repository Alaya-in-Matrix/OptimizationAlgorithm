#pragma once
#include<random>
// #define DEBUG_OPTIMIZER
#define WRITE_LOG

#ifndef RAND_SEED
#ifdef DEBUG_OPTIMIZER
#define RAND_SEED 55
#else
#define RAND_SEED std::random_device{}()
#endif
#endif

