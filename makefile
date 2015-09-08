all:
	g++ -std=c++14 -march=native -fopenmp -Ofast -fno-finite-math-only main.cpp -o output/normalfet -lblas -lgomp -lsuperlu
	cp strip.sh usage.txt output/
