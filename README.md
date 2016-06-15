# 优化算法实现报告

* Author: lvwenlong_lambda@qq.com
* Last Modified:2016年06月15日 星期三 18时49分12秒 三


## project简介

这个 project 是杜建洪老师课程中介绍的优化算法的 c++ 实现。实现了如下算法：
* 各种一维查找算法，如斐波那契法、黄金分割法、外推法
* 基于 strong wolfe condition 的不精确线搜索方法
* 梯度下降法(Gradient Descent Method)

## 编译与构建

这个project使用**CMake**来管理，使用者需要在系统中事先安装CMake，程序是在ubuntu操作系统下编写与测试，CMake也可以生成Windows下Visual Studio的工程文件，具体的使用请参考CMake手册。

使用者可以使用如下方式来编译：

```bash
cd /path/to/this/project/src
mkdir out_build
cd out_build
cmake ..
make
cd ..
```

在运行`cmake ..`命令时，可以通过下列的两个命令行选项来控制程序的行为:
* `-DWRITE_LOG=ON/OFF`, 是否在优化时对每个点进行记录，如果为`OFF`则只记录最终的最优点
* `-DDEBUG_OPTIMIZER=ON/OFF`, 是否开启debug模式，如果为`ON`,则会使用统一的随机数发生器种子，这样保证每次运行，都得到相同的结果。

## 基本数据结构

这个 project 关注的重点在算法运行的迭代次数。因此，并没对算法运行时采用的数据结构进行优化。

用来表示函数输入参数、函数返回结果、以及待优化函数的数据类型定义如下：

```cpp
// 优化函数输入参数向量
typedef std::vector<double> Paras;
// 优化函数执行结果
class Solution
{
    // Para与evaluated result放在一个class中，方便(partial) sort
    Paras _solution;
    std::vector<double> _violation;  // sum of constraint violation
    double _fom;

public:
    Solution(const Paras& s, const std::vector<double>& cv, double fom) noexcept;
    Solution() =delete;
    double fom() const noexcept;
    double sum_violation() const noexcept;
    const std::vector<double>& violations() const noexcept;
    const Paras& solution() const noexcept;
    Solution& operator=(const Solution&) =default;
    bool operator<(const Solution& s) const noexcept { return _fom < s.fom(); }
    bool operator<=(const Solution& s) const noexcept { return _fom <= s.fom(); }
};
typedef std::function<Solution(const Paras&)> ObjFunc;
```
使用`std::vector<double>`来表示待优化函数的输入参数，并将其`typedef`为`Paras`。

将输入参数、目标函数的值fom, 以及约束violation打包成一个class `Solution`, 这样会带来额外的拷贝开销，但好处是编程时更加方便，比如，可以很方便的对一组函数的解进行
排序，选出最好或者最差的解，如果把输入参数跟目标函数输出分开存储，则如果要对目标函数的解进行排序，则需要额外处理输入参数与目标函数输出的同步问题。

对于目标函数的表示，我采用了c++11中函数式编程的特性。在c++11中，可以用 lambda expression 来表示一个函数，这样表示的函数可以作为数据处理，可以作为另一个函数的输入参
数，也可以作为一个函数的返回值。在这个 project 中，目标函数表示为一个输入为`const Paras&`，输出类型为`Solution`的函数。这个函数由用户定义，并作为 optimizer 的构造
函数的一个参数。

## 一维优化算法

## 不精确线搜索

## 多元函数优化算法
