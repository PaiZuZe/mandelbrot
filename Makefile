CFLAGS=-Wall -Wextra -pedantic -lpthread `libpng-config --ldflags`

file: mandelbrot.cpp
	g++ mandelbrot.cpp $(CFLAGS) -g -O2 -o mbrot

blob:
	g++ mandelbrot.cpp -O2 -o mbrot
test:
	./mbrot -2.0 -2.0 2.9 2.0 1920 1080 CPU 4 bob.png
clean:
	rm mbrot
	rm bob.png
