#include <iostream> //cout, cerr
#include <complex>
#include <string> //stof
#include <vector>

#include <png++/png.hpp>

#define INTER_LIMIT 200

#define DIE(...) { \
        std::cerr << __VA_ARGS__; \
        std::exit (EXIT_FAILURE); \
}

int get_inter(std::complex<float> c) {
    std::complex<float> z(0.0, 0.0);
    int i;
    for (i = 0; i < INTER_LIMIT || std::norm(z) > 4; i++) {
        z = std::pow(z, 2) + c;
    }
    return i;
}

std::vector<std::vector<int>> sinlge_mandel(std::complex<float> c0, const float del_y, const float del_x, const int w, const int h){
    std::vector<std::vector<int>> res_matrix(w + 1, std::vector<int>(h + 1));
    std::complex<float> del(0, 0);
    for (int i = 0; i <= h; i += 1) {
        for (int j = 0; j <= w; j += 1) {
            res_matrix[i][j] = get_inter(c0 + del);
            del.imag(del.imag() + del_y);
        }
        del.real(del.real() + del_x);
        del.imag(0);
    }
    return res_matrix;
}

void create_picture(std::vector<std::vector<int>> matrix, const std::string file_name, const int w, const int  h) {
    png::image< png::rgb_pixel > image(w, h);
    for (png::uint_32 y = 0; y < image.get_height(); ++y) {
        for (png::uint_32 x = 0; x < image.get_width(); ++x) {
            image[y][x] = png::rgb_pixel(128,0,128);
        }
    }
    for (auto i : matrix) {
        for (auto j: i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
    image.write(file_name);

}

int main(int argc, char** argv) {
    if (argc != 10) {
        DIE("Wrong number of arguments\n");
    }
    std::complex<float> c0(std::stof(argv[1]), std::stof(argv[2]));
    std::complex<float> c1(std::stof(argv[3]), std::stof(argv[4]));
    const int w = std::stoi(argv[5]);
    const int h = std::stoi(argv[6]);
    const float del_x = (c1.real() - c0.real()) / w;
    const float del_y = (c1.imag() - c0.imag()) / h;
    /*
    const int num_threads = std::stoi(argv[8]);
    const std::string comp_flag = argv[7];
    */
    const std::string file_name = argv[9];
    std::vector<std::vector<int>> res;

    res = sinlge_mandel(c0, del_y, del_x, w, h);
    create_picture(res, file_name, w, h);

    return 0;
}