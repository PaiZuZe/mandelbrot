#include <iostream> //cout, cerr
#include <complex>
#include <string> //stof
#include <vector>

#define INTER_LIMIT 200

#define DIE(...) { \
        std::cerr << __VA_ARGS__; \
        std::exit (EXIT_FAILURE); \
}

int get_inter(std::complex<float> c) {
    std::complex<float> z(0.0, 0.0);
    int i;
    for (i = 0; i < INTER_LIMIT; i++) {
        if (std::abs(z) > 2) {
            break;
        }
        z = std::pow(z, 2) + c;
    }
    return i;
}

std::vector<std::vector<int>> sinlge_mandel(std::complex<float> c0, std::complex<float> c1, const int del_y, const int del_x, const int w, const int h){
    std::vector<std::vector<int>> res_matrix;
    std::complex<float> del(del_x, del_y);
    for (int i = 0; i < w; i += del_x) {
        res_matrix.push_back(std::vector<int> ());
        for (int j = 0; j < h; j += del_y) {
            del.imag(del.imag() + del_y);
            res_matrix[i].push_back(get_inter(c0 + del));

        }
        del.real(del.real() + del_x);
        del.imag(del_y);
    }
    return res_matrix;
}

int main(int argc, char** argv) {
    if (argc != 10) {
        DIE("Wrong number of arguments\n");
    }
    std::complex<float> c0(std::stof(argv[1]), std::stof(argv[2]));
    std::complex<float> c1(std::stof(argv[3]), std::stof(argv[4]));
    const int w = std::stoi(argv[5]);
    const int h = std::stoi(argv[6]);
    const int del_x = (w * 0.01 > 1) ? w * 0.01 : 1;  //WRONG ?????
    const int del_y = (h * 0.01 > 1) ? h * 0.01 : 1;  //WRONG ?????
    const int num_threads = std::stoi(argv[8]);
    const std::string comp_flag = argv[7];
    const std::string file_name = argv[9];

    sinlge_mandel(c0, c1, del_y, del_x, w, h);

    return 0;
}