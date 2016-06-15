# 优化算法实现报告

* Author: lvwenlong_lambda@qq.com
* Last Modified:2016年06月15日 星期三 23时39分20秒 三


## 1. project 简介 

这个 project 是杜建洪老师课程中介绍的优化算法的 c++ 实现。实现了如下算法：
* 各种一维查找算法，如斐波那契法、黄金分割法、外推法
* 基于 strong wolfe condition 的不精确线搜索方法
* 梯度下降法(Gradient Descent Method)

## 2. 编译与构建

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

在运行`cmake ..`命令时，可以通过下列的两个命令行选项来控制程序的行为：
* `-DWRITE_LOG=ON/OFF`, 是否在优化时对每个点进行记录，如果为`OFF`则只记录最终的最优点
* `-DDEBUG_OPTIMIZER=ON/OFF`, 是否开启debug模式，如果为`ON`,则会使用统一的随机数发生器种子，这样保证每次运行，都得到相同的结果。

## 3. 基本数据结构

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

对于目标函数的表示，我采用了c\+\+11中函数式编程的特性。在c\+\+11中，可以用 lambda expression 来表示一个函数，这样表示的函数可以作为数据处理，可以作为另一个函数的输入参
数，也可以作为一个函数的返回值。在这个 project 中，目标函数表示为一个输入为`const Paras&`，输出类型为`Solution`的函数。这个函数由用户定义，并作为 optimizer 的构造
函数的一个参数。

## 4. 一维优化算法

一元函数优化是优化算法的基本，即使是多元函数, 在确定了下一步搜索方向之后，也往往在搜索方向上进行线搜索(line search)，在这个 project 中，实现了 Fibonacci 法, 黄金分割
法和外推内插法这三个优化算法。

首先定义一维函数优化的基类:

```cpp
class Optimizer1D
{
protected:
    ObjFunc _func;

public:
    Optimizer1D(ObjFunc func) noexcept : _func(func) {}
    virtual Solution optimize() noexcept = 0;
};
```

`Optimizer1D` 的构造函数接受一个 `ObjFunc` 类型，`ObjFunc` 即上一节介绍过的表示目标函数的类型，这里的目标函数必须是一元函数，否则，程序可能会出错。

`Optimizer1D::optimize()`是一个纯虚类，所有继承 `Optimizer1D` 类的派生类都需要实现这个方法，具体的一维优化算法就实现在这里。

#### 4.1 Fibonacci 法

Fibonacci 法的类型声明如下:
```cpp
class FibOptimizer : public Optimizer1D
{
    const double _lb;
    const double _ub;
    const size_t _iter;

public:
    FibOptimizer(ObjFunc f, double lb, double ub, size_t iter = 16) noexcept;
    Solution optimize() noexcept;
    ~FibOptimizer() {}
};
```
Fibonacci 法需要提供一个一元目标函数，同时，需要提供搜索的下界与上界，Fibonacci最终的精度随迭代次数指数下降，因此还需要提供一个迭代次数，设置迭代次数默认为16。

Fibonacci 法优化算法实现如下：
```cpp
Solution FibOptimizer::optimize() noexcept
{
    // 1-D function
    // function shoulde be convex function
    double a1 = _lb;
    double a2 = _ub;
    
    // ensure obj function is 1D function
    if (a1 > a2)
    {
        cerr << ("Range is [" + to_string(a1) + ", " + to_string(a2) + "]") << endl;
        exit(EXIT_FAILURE);
    }

    // pre-calculate fibonacci list
    vector<double> fib_list{1, 1};
    if (_iter > 2)
        for (size_t i = 2; i < _iter; ++i) fib_list.push_back(fib_list[i - 1] + fib_list[i - 2]);

    double y1, y2;
    for (size_t i = _iter - 1; i > 0; --i)
    {
        const double rate = fib_list[i - 1] / fib_list[i];
        const double interv_len = a2 - a1;
        const double a3 = a2 - rate * interv_len;
        const double a4 = a1 + rate * interv_len;
        assert(a3 <= a4);
        const double y3 = _func({a3}).fom();
        const double y4 = _func({a4}).fom();

        if (y3 < y4)
        {
            a2 = a4;
            y2 = y4;
        }
        else
        {
            a1 = a3;
            y1 = y3;
        }
    }
    return _func({a1});
}
```
