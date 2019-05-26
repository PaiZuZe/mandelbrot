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

__global__ void fill_matrix (int *res, const int w, const int h, thrust::complex<float> c0, const float del_y, const float del_x, const int threads, const int blocks, const int offset) {
    thrust::complex<float> del(0, 0);
    int k = threadIdx.x + blockIdx.x*threads + blocks*threads*offset;
    if (k >= w*h)
        return;
    del.real(del_x * (k%w));
    del.imag(del_y * (k/w));
    res[k] = get_inter(c0 + del);
    return;
}

__host__ void prepare (int *res_matrix, const int w, const int h, thrust::complex<float> c0, const float del_y, const float del_x, const int threads) {
    int *d_res_matrix; 
    int *d_w; 
    int *d_h;
    thrust::complex<float> *d_c0; 
    float *d_del_y; 
    float *d_del_x; 
    
    cudaSetDevice(0);

    cudaMallocManaged((void **) &d_res_matrix, sizeof(int)*w*h);
    cudaMallocManaged((void **) &d_w, sizeof(int));
    cudaMallocManaged((void **) &d_h, sizeof(int));
    cudaMallocManaged((void **) &d_c0, sizeof(thrust::complex<float>));
    cudaMallocManaged((void **) &d_del_y, sizeof(float));
    cudaMallocManaged((void **) &d_del_x, sizeof(float));
    
    cudaMemcpy(d_w, &w, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &h, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c0, &c0, sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_del_y, &del_y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_del_x, &del_x, sizeof(float), cudaMemcpyHostToDevice);
    
    int block = 1024;
    int max = ((w*h) / (threads*block)) + 1;
    for (int i = 0; i < max; ++i) {
        fill_matrix<<<block, threads>>> (d_res_matrix, *d_w, *d_h, *d_c0, *d_del_y, *d_del_x, threads, block, i);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(res_matrix, d_res_matrix, sizeof(int)*w*h, cudaMemcpyDeviceToHost);
    
    cudaFree(d_res_matrix);
    cudaFree(d_w);
    cudaFree(d_h);
    cudaFree(d_c0);
    cudaFree(d_del_y);
    cudaFree(d_del_x);
    return;
}
