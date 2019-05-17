#ifndef MANDEL_H 
#define MANDEL_H

#include <thrust/device_vector.h>
#include <thrust/complex.h>
#define INTER_LIMIT 255


__host__ void prepare (thrust::device_vector<float> res_matrix, thrust::complex<float> c0, const float del_y, const float del_x, const int threads);

#endif 