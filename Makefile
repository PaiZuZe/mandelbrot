CFLAGS=-Wall -Wextra -pedantic -fopenmp -lpthread `libpng-config --ldflags`
NVCC=nvcc
CC=g++

mandel.o: mandel.cu
	$(NVCC) mandel.cu

all: mandel.o
	$(CC) mandelbrot.cpp mandel.o $(CFLAGS) -g -O2 -o mbrot

.PHONY: test
test:
	./mbrot -2.0 -2.0 2.9 2.0 1920 1080 CPU 4 bob.png
.PHONY: clean
clean:
	rm mbrot
	rm bob.png
