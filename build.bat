@echo off
pushd cpp
g++ -O2 -Iarmadillo/include -c -o AdaptiveRobustRegression.o AdaptiveRobustRegression.cpp
g++ -O2 -shared -o AdaptiveRobustRegression.dll -L. AdaptiveRobustRegression.o -Larmadillo/lib -lopenblas
popd
echo Done
