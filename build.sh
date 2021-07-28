#!/bin/sh
OUT="../pyAutoAdaptiveRobustRegression/bin/macos"
pushd cpp
mkdir -p "${OUT}"
clang++ -std=c++17 -O2 -Iarmadillo/include -c -o "${OUT}/pyAutoAdaptiveRobustRegression.o" pyAutoAdaptiveRobustRegression.cpp
clang++ -O2 -shared -o "${OUT}/pyAutoAdaptiveRobustRegression.dylib" -L. "${OUT}/pyAutoAdaptiveRobustRegression.o" -L/usr/local/opt/openblas/lib -lopenblas
popd
echo Done