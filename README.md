# README

* Author: lvwenlong_lambda@qq.com
* Last Modified:2016年06月16日 星期四 18时52分21秒 四

## About this project

* Some Numerical Optimization algorithm I learned in my graduate student course *RF Circuit Design and Optimization*

## Algorithms:

* 1D function optimization
* Gradient Descent Method
* Conjugate Gradient Method
* Newton's Method
* Quasi-Newton Methos: BFGS and DFP
* Simplex Method
* Powell's Method

## Compile and Build

I use CMake to build this project, you migit do out-of-source build simply by:

```bash
mkdir /path/to/build/dir
cd /path/to/build/dir
cmake /path/to/project/src
make
```

Two options are used to control the program's behaviour:

* `-DWRITE_LOG=ON/OFF`, if this option is `ON`, then every objective function evaluation would be recorded into the corresponding log file, otherwise, only the final optimized point would be recorded.
* `-DDEBUG_OPTIMIZER=ON/OFF`, if this option is `ON`, then the seed for random engine defined in `def.h` would be used, otherwise `std::random_device` would be used.

## Report

A report written in Chinese is given in directory report/
