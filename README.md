# pyAutoAdaptiveRobustRegression

**Auto Adaptive Robust Regression Python Package**

## Description

This python package implements the Adaptive Gradient Descent, Adaptive Gradient Descent with Barzilai-Borwein Method and Adaptive Gradient Descent with Backtracking Method. It also includes the Huber Mean Estimation, Huber Covariance Matrix Estimation, Huber Regression and Adaptive Huber Regression from `R` library [FarmTest](https://CRAN.R-project.org/package=FarmTest), written by [Xiaoou Pan](https://www.math.ucsd.edu/~xip024/).

## Installation

This python package can be installed on **Windows**, **Mac** and **Linux**.

Install `pyAutoAdaptiveRobustRegression` with `pip`:

```
pip install pyAutoAdaptiveRobustRegression
```

## Requirements on Operating Systems

For Windows:

There is no requirement for Windows. The armadillo and openblas libraries have already included.

For Mac:

```
brew install armadillo openblas
```

For Linux:

```
apt install armadillo openblas
```

## Functions

There are seven functions in this package:

-   `agd`: Adaptive Gradient Descent
-   `agd_bb`: Adaptive Gradient Descent with Barzilai-Borwein Method
-   `agd_backtracking`: Adaptive Gradient Descent with Backtracking Method
-   `huber_mean`: Huber Mean Estimation
-   `huber_cov`: Huber Covariance Matrix Estimation
-   `huber_reg`: Huber Regression
-   `ada_huber_reg`: Adaptive Huber Regression

## License

MIT

## Author(s)

Yichi Zhang <yichi.zhang@worc.ox.ac.uk>, Qiang Sun <qiang.sun@utoronto.ca>

## References

Sun, Q. (2021). Do we need to estimate the variance in robust mean estimation? [Paper](https://arxiv.org/pdf/2107.00118.pdf) 

Bose, K., Fan, J., Ke, Y., Pan, X. and Zhou, W.-X. (2020). FarmTest: An R package for factor-adjusted robust multiple testing. *R. J.* **12** 372-387. [Paper](https://journal.r-project.org/archive/2021/RJ-2021-023/index.html)

Fan, J., Ke, Y., Sun, Q. and Zhou, W.-X. (2019). FarmTest: Factor-adjusted robust multiple testing with approximate false discovery control. *J. Amer. Statist. Assoc.* **114** 1880-1893. [Paper](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1527700) 

Sun, Q., Zhou, W.-X. and Fan, J. (2020). Adaptive Huber regression. *J. Amer. Stat. Assoc.* **115** 254-265. [Paper](https://doi.org/10.1080/01621459.2018.1543124)

Wang, L., Zheng, C., Zhou, W. and Zhou, W.-X. (2020). A new principle for tuning-free Huber regression. *Stat. Sinica* to appear. [Paper](https://www.math.ucsd.edu/~wez243/tfHuber.pdf)