all: AdaptiveRobustRegression.so

AdaptiveRobustRegression.o: AdaptiveRobustRegression.cpp
	g++ -fPIC  -O2 -Iarmadillo/include -c -o AdaptiveRobustRegression.o AdaptiveRobustRegression.cpp

AdaptiveRobustRegression.so: AdaptiveRobustRegression.o
	 g++  -O2 -shared -o AdaptiveRobustRegression.so  -L. AdaptiveRobustRegression.o -Larmadillo/lib -lopenblas

clean:
	rm -f AdaptiveRobustRegression.so AdaptiveRobustRegression.o
