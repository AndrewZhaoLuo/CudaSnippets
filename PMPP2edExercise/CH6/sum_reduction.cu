#include <cuda.h>
#include <iomanip>
#include <iostream>

// Tolerance for comparing floating points
#define EPSILON 0.0001

#define DEBUG false 

// Example of sum reduction kernel, exercise at end of ch6 of PMPP 2ed
// Example sum reduction from fig. 6.2 of PMPP 2ed
__global__
void sum_reduction_fig6_2(float* d_input_arr, float* d_out, int n) {
    extern __shared__ float partialSum[]; // dynamic shared memory

    // load memory into partialSum
    unsigned int t = threadIdx.x;
    partialSum[t] = d_input_arr[t];

    if (DEBUG) {
        printf("Thread %d loads: %f\n", t, partialSum[t]);
    }

    // run algorithm
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0) partialSum[t] += partialSum[t + stride];
    }

    __syncthreads();
    *d_out = partialSum[0];
}

void sum_reduction_cpu(float* d_input_arr, float* out, int n) {
    float result = 0;
    for (int i = 0; i < n; i++) {
        result += d_input_arr[i];
    }
    *out = result;
}

void run_cuda_impl(float* h_arr, float* h_out, int n) {
    float* d_arr, *d_out;
    cudaMalloc((void**)&d_arr, sizeof(float) * n);
    cudaMalloc((void**)&d_out, sizeof(float));
    cudaMemcpy(d_arr, h_arr, sizeof(float) * n, cudaMemcpyHostToDevice);

    // call kernel
    // TODO: handle bigger arrays -- split up blocks
    sum_reduction_fig6_2<<<1, n, sizeof(float) * n>>>(d_arr, d_out, n);

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_out);
}

int main() {
    int N = 1024;
    float* h_arr = (float*) malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 10;
    }

    float result_cpu;
    sum_reduction_cpu(h_arr, &result_cpu, N);
    std::cout << "CPU result : " << result_cpu << std::endl;

    float result_cuda;
    run_cuda_impl(h_arr, &result_cuda, N);
    std::cout << "CUDA result: " << result_cuda << std::endl;

    return 0;
}