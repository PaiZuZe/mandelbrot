CFLAGS=-Wall -Wextra -pedantic -fopenmp -L/usr/local/cuda/lib64 -lcuda -lcudart -lpthread `libpng-config --ldflags`
NVCC=nvcc
CC=g++

mandel.o: mandel.cu
	$(NVCC) mandel.cu -c

all: mandel.o
	$(CC) mandelbrot.cpp mandel.o $(CFLAGS) -g -O2 -o mbrot

.PHONY: cpu
cpu:
	./mbrot -2.0 -2.0 2.9 2.0 1920 1080 CPU 4 bob.png

.PHONY: gpu
gpu:
	./mbrot -2.0 -2.0 2.9 2.0 1920 1080 GPU 4 bob.png


.PHONY: clean
clean:
	rm mbrot
	rm bob.png
	rm *.o
