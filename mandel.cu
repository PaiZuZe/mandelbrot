#include "mandel.cuh"

__device__ int get_inter (thrust::complex<float> c) {
    int i;
    thrust::complex<float> z(0.0, 0.0);

    for (i = 0; i< INTER_LIMIT; ++i) {
        if (thrust::norm(z) > 4 ) {
            break;
        }
        z = thrust::pow(z, 2) + c;
    }
    return i;
}

__global__ void fill_matrix (int *res_matrix, const int w, const int h, thrust::complex<float> c0, const float del_y, const float del_x) {
    thrust::complex<float> del(0, 0);
    for (int i = 0; i < h; i += 1) {
        for (int j = 0; j < w; j += 1) {
            del.real(del_x * j);
            del.imag(del_y * i);
            res_matrix[i*w + j] = get_inter(c0 + del);
        }
    }
    return;
}

__host__ void prepare (int *res_matrix, const int w, const int h, thrust::complex<float> c0, const float del_y, const float del_x, const int threads) {
    int *d_res_matrix; 
    int *d_w; 
    int *d_h;
    thrust::complex<float> *c_c0; 
    float *d_del_y; 
    float *d_del_x; 
    cudaMalloc((void **) &d_res_matrix, sizeof(int)*w*h);
    cudaMalloc((void **) &d_w, sizeof(int));
    cudaMalloc((void **) &d_h, sizeof(int));
    cudaMalloc((void **) &c_c0, sizeof(thrust::complex<float>));
    cudaMalloc((void **) &d_del_y, sizeof(float));
    cudaMalloc((void **) &d_del_x, sizeof(float));
    cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_c0, &c0, sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_del_y, &del_y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_del_x, &del_x, sizeof(float), cudaMemcpyHostToDevice);
    //fill_matrix<<<1, 1>>> (d_res_matrix, d_w, d_h, d_c0, d_del_y, d_del_x);
    return;
}