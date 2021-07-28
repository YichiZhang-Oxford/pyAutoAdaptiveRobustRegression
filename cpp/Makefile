all: pyAutoAdaptiveRobustRegression.so

pyAutoAdaptiveRobustRegression.o: pyAutoAdaptiveRobustRegression.cpp
	g++ -fPIC  -O2 -Iarmadillo/include -c -o pyAutoAdaptiveRobustRegression.o pyAutoAdaptiveRobustRegression.cpp

pyAutoAdaptiveRobustRegression.so: pyAutoAdaptiveRobustRegression.o
	 g++  -O2 -shared -o pyAutoAdaptiveRobustRegression.so  -L. pyAutoAdaptiveRobustRegression.o -Larmadillo/lib -lopenblas

clean:
	rm -f pyAutoAdaptiveRobustRegression.so pyAutoAdaptiveRobustRegression.o
