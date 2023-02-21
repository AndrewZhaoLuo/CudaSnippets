/*
Playing around with implementing layout transform / transposition efficiently
*/
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <stdio.h>

// TODO: monitor for bank conflicts
// Support FP16 datatype
// Compare bank conflicts for FP32 vs FP16
// Do shit...

template <typename T, int copy_loads_per_thread>
__global__ void CudaBaselineCopy(T *mat1, T *mat2) {
  int start_index_t0 = blockIdx.x * blockDim.x * copy_loads_per_thread;
  for (int i = 0; i < copy_loads_per_thread; i++) {
    int index_t0 = start_index_t0 + i * blockDim.x;
    mat2[index_t0 + threadIdx.x] = mat1[index_t0 + threadIdx.x];
  }
}

template <typename T, int copy_loads_per_thread, int num_threads_per_block,
          int rows, int cols>
void CudaBaselineCopyWrapped(T *d_mat1, T *d_mat2) {
  int num_blocks =
      (rows * cols) / (copy_loads_per_thread * num_threads_per_block);
  CudaBaselineCopy<T, copy_loads_per_thread>
      <<<num_blocks, num_threads_per_block>>>(d_mat1, d_mat2);
}

template <typename T, int rows, int cols, int loads_per_thread,
          int num_threads_per_block>
__global__ void CudaTransposeNaive(T *mat, T *matT) {
  int y_start = blockIdx.y * loads_per_thread;
  int x_start = blockIdx.x * num_threads_per_block + threadIdx.x;
  for (int i = 0; i < loads_per_thread; i++) {
    matT[x_start * rows + (y_start + i)] = mat[(y_start + i) * cols + x_start];
  }
}

template <typename T, int rows, int cols, int tile_size, int padding_shared>
__global__ void CudaTransposeSharedIntermediate(T *mat, T *matT) {
  // num threads = tile_size each thread will load and thread tile_size
  // elements.
  __shared__ float tile[tile_size][tile_size + padding_shared];

  // index in mat
  int y_start = blockIdx.y * tile_size;
  int x_start = blockIdx.x * tile_size + threadIdx.x;

  // cooperatively load shared memory, do not transpose yet
  for (int i = 0; i < tile_size; i++) {
    tile[i][threadIdx.x] = mat[(y_start + i) * cols + x_start];
  }

  __syncthreads();

  // index in matT
  x_start = blockIdx.y * tile_size + threadIdx.x;
  y_start = blockIdx.x * tile_size;
  for (int i = 0; i < tile_size; i++) {
    // matT[x_start * rows + y_start + i] = tile[i][threadIdx.x];
    matT[(y_start + i) * rows + x_start] = tile[threadIdx.x][i];
  }
}

template <typename T, int rows, int cols, int loads_per_thread,
          int num_threads_per_block>
void CudaTransposeNaiveWrapped(T *d_mat, T *d_matT) {
  int num_blocks_y = rows / loads_per_thread;
  assert(rows % loads_per_thread == 0);
  int num_blocks_x = cols / num_threads_per_block;
  assert(cols % num_threads_per_block == 0);
  dim3 dimBlock(num_blocks_x, num_blocks_y, 1);
  dim3 dimGrid(num_threads_per_block, 1, 1);
  CudaTransposeNaive<T, rows, cols, loads_per_thread, num_threads_per_block>
      <<<dimBlock, dimGrid>>>(d_mat, d_matT);
}

template <typename T, int rows, int cols, int tile_size, int padding_shared>
void CudaTransposeIntermediateWrapped(T *d_mat, T *d_matT) {
  int num_blocks_y = rows / tile_size;
  assert(rows % tile_size == 0);
  int num_blocks_x = cols / tile_size;
  assert(cols % tile_size == 0);
  dim3 dimBlock(num_blocks_x, num_blocks_y, 1);
  dim3 dimGrid(tile_size, 1, 1);
  CudaTransposeSharedIntermediate<T, rows, cols, tile_size, padding_shared>
      <<<dimBlock, dimGrid>>>(d_mat, d_matT);
}

template <typename T, int rows, int cols>
void doTransposeCUDA(T *h_mat, T *h_matT, int mode) {
  int matrix_size = rows * cols * sizeof(T);
  T *d_mat, *d_matT;

  cudaMalloc((void **)&d_mat, matrix_size);
  cudaMalloc((void **)&d_matT, matrix_size);
  cudaMemcpy(d_mat, h_mat, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matT, h_matT, matrix_size, cudaMemcpyHostToDevice);

  // Call kernel
  switch (mode) {
  case 0:
    CudaTransposeNaiveWrapped<T, rows, cols, 4, 1024>(d_mat, d_matT);
    break;
  case 1:
    CudaTransposeIntermediateWrapped<T, rows, cols, 32, 0>(d_mat, d_matT);
    break;
  case 2:
    CudaTransposeIntermediateWrapped<T, rows, cols, 32, 1>(d_mat, d_matT);
    break;
  case 3:

  default:
    printf("Error unknown mode %d\n", mode);
    throw std::invalid_argument("Unknown mode");
  }
  cudaMemcpy(h_matT, d_matT, matrix_size, cudaMemcpyDeviceToHost);
  cudaFree(d_mat);
  cudaFree(d_matT);
}

// Only support float type
template <typename T, int rows, int cols>
void doTransposeCPU(T *h_mat, T *h_matT) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      h_matT[c * rows + r] = h_mat[r * cols + c];
    }
  }
}

template <typename T, int matrix_rows, int matrix_cols>
void runTransposeExperiment(int mode) {
  T *h_mat = (T *)malloc(sizeof(T) * matrix_rows * matrix_cols);
  T *h_matT_experimental = (T *)malloc(sizeof(T) * matrix_rows * matrix_cols);
  T *h_matT_control = (T *)malloc(sizeof(T) * matrix_rows * matrix_cols);

  for (int i = 0; i < matrix_rows * matrix_cols; i++) {
    h_mat[i] = rand() % 100;
    h_matT_experimental[i] = 0;
    h_matT_control[i] = 0;
  }

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  doTransposeCUDA<T, matrix_rows, matrix_cols>(h_mat, h_matT_experimental,
                                               mode);
  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CUDA CPU time %.5f (s)\n", cpu_time_used);

  start = clock();
  doTransposeCPU<T, matrix_rows, matrix_cols>(h_mat, h_matT_control);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CPU time %.5f (s)\n", cpu_time_used);

  char *h_mat1_check = (char *)h_matT_experimental;
  char *h_mat2_check = (char *)h_matT_control;
  for (int i = 0; i < matrix_rows * matrix_cols * sizeof(T); i++) {
    assert(h_mat1_check[i] == h_mat2_check[i]);
  }

  free(h_mat);
  free(h_matT_experimental);
  free(h_matT_control);
}

template <typename T, int matrix_rows, int matrix_cols,
          int copy_loads_per_thread, int num_threads_per_block>
void runCopyThroughputExperiment() {
  int matrix_size_bytes = sizeof(T) * matrix_cols * matrix_rows;
  T *h_mat1 = (T *)malloc(sizeof(T) * matrix_size_bytes);
  T *h_mat2 = (T *)malloc(sizeof(T) * matrix_size_bytes);

  for (int i = 0; i < matrix_rows * matrix_cols; i++) {
    h_mat1[i] = rand() % 100;
    h_mat2[i] = 1;
  }

  T *d_mat1, *d_mat2;
  cudaMalloc((void **)&d_mat1, matrix_size_bytes);
  cudaMalloc((void **)&d_mat2, matrix_size_bytes);
  cudaMemcpy(d_mat1, h_mat1, matrix_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat2, h_mat2, matrix_size_bytes, cudaMemcpyHostToDevice);

  // Call kernel
  CudaBaselineCopyWrapped<T, copy_loads_per_thread, num_threads_per_block,
                          matrix_rows, matrix_cols>(d_mat1, d_mat2);

  cudaMemcpy(h_mat2, d_mat2, matrix_size_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_mat1);
  cudaFree(d_mat2);

  char *h_mat1_check = (char *)h_mat1;
  char *h_mat2_check = (char *)h_mat2;
  for (int i = 0; i < matrix_rows * matrix_cols * sizeof(T); i++) {
    assert(h_mat1_check[i] == h_mat2_check[i]);
  }
}

#define DTYPE __half
int main(int argc, char **argv) {
  runCopyThroughputExperiment<DTYPE, 1024, 1024 * 2, 4, 256>();
  runTransposeExperiment<DTYPE, 1024, 1024 * 2>(0);
  runTransposeExperiment<DTYPE, 1024, 1024 * 2>(1);
  runTransposeExperiment<DTYPE, 1024, 1024 * 2>(2);
  return 0;
}