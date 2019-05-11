#include <iostream> //cout, cerr
#include <complex>
#include <string> //stof
#include <vector>

#include <png++/png.hpp>

#define INTER_LIMIT 255

#define DIE(...) { \
        std::cerr << __VA_ARGS__; \
        std::exit (EXIT_FAILURE); \
}

/* For debugging onlt, remove before delivery
*/
void print(std::vector<std::vector<int>> &res_matrix) {
    for (auto i : res_matrix) {
        for (auto j : i) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
    return;
}

int get_inter(std::complex<float> c) {
    std::complex<float> z(0.0, 0.0);
    int i;
    for (i = 0; i < INTER_LIMIT; i++) {
        if (std::norm(z) > 4) {
            break;
        }
        z = std::pow(z, 2) + c;
    }
    return i;
}

void sinlge_mandel(std::vector<std::vector<int>> &res_matrix, std::complex<float> c0, const float del_y, const float del_x, const int w, const int h){
    std::complex<float> del(0, 0);
    for (int i = 0; i < h; i += 1) {
        for (int j = 0; j < w; j += 1) {
            del.real(del_x * j);
            del.imag(del_y * i);
            res_matrix[i][j] = get_inter(c0 + del);
        }
    }
    return;
}

void create_picture(std::vector<std::vector<int>> &matrix, const std::string file_name, const int w, const int  h) {
    png::image< png::rgb_pixel > image(w, h);
    for (png::uint_32 i = 0; i < image.get_height(); ++i) {
        for (png::uint_32 j = 0; j < image.get_width(); ++j) {
            image[i][j] = png::rgb_pixel(matrix[i][j], matrix[i][j], matrix[i][j]);
        }
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
    const float del_x = (c1.real() - c0.real()) / (w - 1);
    const float del_y = (c1.imag() - c0.imag()) / (h - 1);
    /*
    const int num_threads = std::stoi(argv[8]);
    const std::string comp_flag = argv[7];
    */
    const std::string file_name = argv[9];
    std::vector<std::vector<int>> res;
    res.resize(h);
    for (int i = 0; i < h; ++i) {
        res[i].resize(w);
    }
    sinlge_mandel(res, c0, del_y, del_x, w, h);
    create_picture(res, file_name, w, h);

    return 0;
}