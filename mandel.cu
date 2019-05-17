__device__ int get_interger (thrust::complex<float> c) {
    int i;
    for (i = 0; i< INTER_LIMIT; ++i) {
        if (thrust::norm(z) > 4 ) {
            break;
        }
        z = thrust::pow(z, 2) + c;
    }
    return i;
}

__global__ void fill_matrix (thrust::device_vector<float> res_matrix, thrust::complex<float> c0, const float del_y, const float del_x, const int threads) {
    thrust::complex<float> del(0, 0);
    const int w = res_matrix[0].size();
    const int h = res_matrix.size();
    for (int i = 0; i < h; i += 1) {
        for (int j = 0; j < w; j += 1) {
            del.real(del_x * j);
            del.imag(del_y * i);
            res_matrix[i][j] = get_inter(c0 + del);
        }
    }
    return;
}

__host__ void prepare (thrust::device_vector<float> res_matrix, thrust::complex<float> c0, const float del_y, const float del_x, const int threads) {
    std::cout << res_matrix.size() << std::endl;
    std::cout << num_threads << std::endl;
    std::cout << c0 << std::endl;
    std::cout << del_y << std::endl;
    std::cout << del_x << std::endl;

    //fill_matrix<<<1, 1>>> (c_res_matrix, c_c0, c_del_y, c_del_x, c_threads);
    return 0;
}