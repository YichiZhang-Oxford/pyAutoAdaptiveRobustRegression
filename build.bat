@echo off
set OUT=../pyAutoAdaptiveRobustRegression/bin/win32
pushd cpp
g++ -O2 -Iarmadillo/include -c -o %OUT%/pyAutoAdaptiveRobustRegression.o pyAutoAdaptiveRobustRegression.cpp
g++ -O2 -shared -o %OUT%/pyAutoAdaptiveRobustRegression.dll -L. %OUT%/pyAutoAdaptiveRobustRegression.o -Larmadillo/lib -lopenblas
popd
echo Done
