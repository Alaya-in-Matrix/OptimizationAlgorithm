#pragma once
#include<random>
#include <iostream>
// These two macros are controlled by CMake
// #define DEBUG_OPTIMIZER
// #define WRITE_LOG

#ifndef RAND_SEED
#ifdef DEBUG_OPTIMIZER
// the value of this macro is irrevalent
#define RAND_SEED 3062469830
#else
#define RAND_SEED std::random_device{}()
#endif
#endif
#define CHECK(COND)                                        \
    {                                                      \
        if (!(COND))                                       \
        {                                                  \
            std::cerr << "Failed: " << #COND << std::endl; \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    }
