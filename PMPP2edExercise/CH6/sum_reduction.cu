#include <assert.h>
#include <cuda.h>
#include <iomanip>
#include <iostream>

// Tolerance for comparing floating points
#define EPSILON 0.0001

#define DEBUG false

const unsigned int MODE_FIG6_2 = 0;
const unsigned int MODE_FIG6_4 = 1;
const unsigned int MODE_FIG6_4_ALT = 2;

// Example of sum reduction kernel, exercise at end of ch6 of PMPP 2ed
// Example sum reduction from fig. 6.2 of PMPP 2ed
__global__ void sum_reduction_fig6_2(float *d_input_arr, float *d_out, int n) {
  extern __shared__ float partialSum[]; // dynamic shared memory

  // load memory into partialSum
  unsigned int t = threadIdx.x;
  partialSum[t] = d_input_arr[t];

  if (DEBUG) {
    printf("Thread %d loads: %f. Block dim x: %d\n", t, partialSum[t], blockDim.x);
  }

  // run algorithm
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (t % (2 * stride) == 0)
      partialSum[t] += partialSum[t + stride];
  }

  __syncthreads();
  *d_out = partialSum[0];
}

// Example sum reduction from fig. 6.4 of PMPP 2ed
__global__ void sum_reduction_fig6_4(float *d_input_arr, float *d_out, int n) {
  extern __shared__ float partialSum[]; // dynamic shared memory

  // load memory into partialSum
  unsigned int t = threadIdx.x;
  partialSum[t] = d_input_arr[t];

  if (DEBUG) {
    printf("Thread %d loads: %f. Block dim x: %d\n", t, partialSum[t], blockDim.x);
  }

  // run algorithm
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      partialSum[t] += partialSum[t + stride];
        if (DEBUG) {
            printf("Thread %d stride: %d curr: %f\n", t, stride, partialSum[t]);
        }
    }
  }

  __syncthreads();
  *d_out = partialSum[0];
}

// Example sum reduction from fig. 6.4 of PMPP 2ed
// This alternate ensure each thread will do some work during kernel execution
// This is done by skipping the first addition phase by having each thread load
// 2 adjacent elements while doing work initially.
__global__ void sum_reduction_fig6_4_alt(float *d_input_arr, float *d_out, int n) {
  extern __shared__ float partialSum[]; // dynamic shared memory

  // load memory into partialSum
  // half as many threads
  unsigned int t = threadIdx.x;
  partialSum[t] = d_input_arr[2 * t] + d_input_arr[2 * t + 1];

  if (DEBUG) {
    printf("Thread %d loads: %f. Block dim x: %d\n", t, partialSum[t], blockDim.x);
  }

  // run algorithm
  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride)
      partialSum[t] += partialSum[t + stride];
  }

  __syncthreads();
  *d_out = partialSum[0];
}

double sum_reduction_cpu(float *d_input_arr, float *out, int n) {
  clock_t start = clock();
  float result = 0;
  for (int i = 0; i < n; i++) {
    result += d_input_arr[i];
  }
  clock_t end = clock();
  *out = result;

  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

double run_cuda_impl(float *h_arr, float *h_out, int n, int mode) {
  float *d_arr, *d_out;
  cudaMalloc((void **)&d_arr, sizeof(float) * n);
  cudaMalloc((void **)&d_out, sizeof(float));
  cudaMemcpy(d_arr, h_arr, sizeof(float) * n, cudaMemcpyHostToDevice);

  // call kernel
  // TODO: handle bigger arrays -- split up blocks
  clock_t start, end;
  double total_time;

  start = clock();
  end = clock();
  switch (mode) {
  case MODE_FIG6_2:
    start = clock();
    sum_reduction_fig6_2<<<1, n, sizeof(float) * n>>>(d_arr, d_out, n);
    break;
  case MODE_FIG6_4:
    start = clock();
    sum_reduction_fig6_4<<<1, n, sizeof(float) * n>>>(d_arr, d_out, n);
    break;
  case MODE_FIG6_4_ALT:
    start = clock();
    sum_reduction_fig6_4_alt<<<1, n / 2, sizeof(float) * n / 2>>>(d_arr, d_out, n); 
    break;
  default:
    assert(false && "Unknown mode!");
  }
  cudaDeviceSynchronize();
  end = clock();
  total_time = ((double)(end - start)) / CLOCKS_PER_SEC;

  cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_arr);
  cudaFree(d_out);
  return total_time;
}

int main() {
  int N = 2048;
  float *h_arr = (float *)malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++) {
    h_arr[i] = rand() % 10;
  }

  if (DEBUG) {
    for (int i = 0; i < N; i++) {
        std::cout << h_arr[i] << " ";
    }
    std::cout << std::endl;
  }

  double total_time;
  float result_cpu;
  total_time = sum_reduction_cpu(h_arr, &result_cpu, N);
  std::cout << "CPU result : " << result_cpu << " time " << 
            std::fixed << std::setprecision(8) << total_time << "(s)"
            << std::endl;

  float result_cuda;
  run_cuda_impl(h_arr, &result_cuda, N, MODE_FIG6_2);
  std::cout << "CUDA 6_2 result: " << result_cuda << " time " << 
            std::fixed << std::setprecision(8) << total_time
            << "(s)" << std::endl;

  run_cuda_impl(h_arr, &result_cuda, N, MODE_FIG6_4);
  std::cout << "CUDA 6_4 result: " << result_cuda << " time " << 
            std::fixed << std::setprecision(8) << total_time
            << "(s)" << std::endl;

  // This modified kernel uses half as many threads and half as much shared memory
  // broadly we can have each thread load N units (adding at the same time)
  // and cut memory usage and threads by N. We can apply a simialr strategy toward
  // the reduction stage too. The only downside is using more registers.
  run_cuda_impl(h_arr, &result_cuda, N, MODE_FIG6_4_ALT);
  std::cout << "CUDA 6_4_alt result: " << result_cuda << " time " << 
            std::fixed << std::setprecision(8) << total_time
            << "(s)" << std::endl;
  return 0;
}