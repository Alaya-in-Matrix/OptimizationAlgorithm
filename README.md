# README

* Author: lvwenlong_lambda@qq.com
* Last Modified:2016年06月01日 星期三 13时58分28秒 三

## About

* 杜建洪老师课程优化算法实现

## Build

I use CMake to build this project, you can just `cd src && cmake .` or you can choose out-of-source build.

Two options are used to control the behaviour of program

* `-DWRITE_LOG=ON/OFF`, whether or not to write record for every point
* `-DDEBUG_OPTIMIZER=ON/OFF`, in the deubg mode, a fixed seed for random number engine is used.
