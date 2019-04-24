CFLAGS=-Wall -Wextra -Werror -pedantic -lpthread

file: mandelbrot.cpp
	g++ mandelbrot.cpp $(CFLAGS) -g -O2 -o mbrot

blob:
	g++ mandelbrot.cpp -O2 -o mbrot
test:
	./mbrot 1.7 4.5 2.3 5.4 100 100 CPU 4 bob.png
clean:
	rm mandelbrot
