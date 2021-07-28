# pyAutoAdaptiveRobustRegression

**Auto Adaptive Robust Regression Python Package**

## Description

This python package implements the Adaptive Gradient Descent, Adaptive Gradient Descent with Barzilai-Borwein Method and Adaptive Gradient Descent with Backtracking Method. It also includes the Huber Mean Estimation, Huber Covariance Matrix Estimation, Huber Regression and Adaptive Huber Regression from `R` library [FarmTest](https://CRAN.R-project.org/package=FarmTest), written by [Xiaoou Pan](https://www.math.ucsd.edu/~xip024/).

## Installation

This python package can be installed on **Windows**, **Mac** and **Linux**.

Install `pyAutoAdaptiveRobustRegression` with `pip`:

```
$ pip install pyAutoAdaptiveRobustRegression
```

## Requirements on Linux and macOS

```sh
apt install armadillo openblas # for Linux
```

```sh
brew install armadillo openblas # for macOS
```

## Functions

There are seven functions in this package:

-   `agd`: Adaptive Gradient Descent
-   `agd_bb`: Adaptive Gradient Descent with Barzilai-Borwein Method
-   `agd_backtracking`: Adaptive Gradient Descent with Backtracking Method
-   `huber_mean`: Huber Mean Estimation
-   `huber_cov`: Huber Covariance Matrix Estimation
-   `huber_reg`: Huber regression.
-   `ada_huber_reg`: Adaptive Huber regression

## License

MIT

## Author(s)

Yichi Zhang <yichi.zhang@worc.ox.ac.uk>, Qiang Sun <qiang.sun@utoronto.ca>
