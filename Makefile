CFLAGS=-Wall -Wextra -Werror -pedantic -lpthread `libpng-config --ldflags`

file: mandelbrot.cpp
	g++ mandelbrot.cpp $(CFLAGS) -g -O2 -o mbrot

blob:
	g++ mandelbrot.cpp -O2 -o mbrot
test:
	./mbrot 0.0 0.0 -0.2 -0.2 5 5 CPU 4 bob.png
clean:
	rm mbrot
