/*
Playing around with implementing layout transform / transposition efficiently
*/
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdexcept>

#define COPY_LOADS_PER_THREAD 1

__global__ void CudaBaselineCopy(float *mat1, float *mat2) {
  int start_index_t0 = blockIdx.x * blockDim.x * COPY_LOADS_PER_THREAD;
  for (int i = 0; i < COPY_LOADS_PER_THREAD; i++) {
    int index_t0 = start_index_t0 + i * blockDim.x;
    mat2[index_t0 + threadIdx.x] = mat1[index_t0 + threadIdx.x];
  }
}

void CudaBaselineCopyWrapped(float *d_mat1, float *d_mat2, int rows, int cols,
                             int num_threads_per_block = 32) {
  int num_blocks = (rows * cols) / (COPY_LOADS_PER_THREAD * num_threads_per_block);
  CudaBaselineCopy<<<num_blocks, num_threads_per_block>>>(
      d_mat1, d_mat2);
}

__global__ void CudaTransposeNaive(float *mat, float *matT, int rows, int cols,
                                   int loads_per_thread,
                                   int num_threads_per_block) {
  int y_start = blockIdx.y * loads_per_thread;
  int x_start = blockIdx.x * num_threads_per_block + threadIdx.x;
  for (int i = 0; i < loads_per_thread; i++) {
    matT[x_start * rows + (y_start + i)] = mat[(y_start + i) * cols + x_start];
  }
}

template<int tile_size=32, int padding_shared=0>
__global__ void CudaTransposeSharedIntermediate(float *mat, float *matT, int rows, int cols) {
  // num threads = tile_size each thread will load and thread tile_size elements.
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

void CudaTransposeNaiveWrapped(float *d_mat, float *d_matT, int rows, int cols,
                               int loads_per_thread = 32,
                               int num_threads_per_block = 32) {
  int num_blocks_y = rows / loads_per_thread;
  assert(rows % loads_per_thread == 0);
  int num_blocks_x = cols / num_threads_per_block;
  assert(cols % num_threads_per_block == 0);
  dim3 dimBlock(num_blocks_x, num_blocks_y, 1);
  dim3 dimGrid(num_threads_per_block, 1, 1);
  CudaTransposeNaive<<<dimBlock, dimGrid>>>(
      d_mat, d_matT, rows, cols, loads_per_thread, num_threads_per_block);
}

template<int tile_size=32, int padding_shared=0>
void CudaTransposeIntermediateWrapped(float *d_mat, float *d_matT, int rows, int cols) {
  int num_blocks_y = rows / tile_size;
  assert(rows % tile_size == 0);
  int num_blocks_x = cols / tile_size;
  assert(cols % tile_size == 0);
  dim3 dimBlock(num_blocks_x, num_blocks_y, 1);
  dim3 dimGrid(tile_size, 1, 1);
  CudaTransposeSharedIntermediate<tile_size, padding_shared><<<dimBlock, dimGrid>>>(
      d_mat, d_matT, rows, cols);
}

void doTransposeCUDA(float *h_mat, float *h_matT, int rows, int cols, int mode=0) {
  int matrix_size = rows * cols * sizeof(float);
  float *d_mat, *d_matT;

  cudaMalloc((void **)&d_mat, matrix_size);
  cudaMalloc((void **)&d_matT, matrix_size);
  cudaMemcpy(d_mat, h_mat, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matT, h_matT, matrix_size, cudaMemcpyHostToDevice);

  // Call kernel
  switch(mode) {
    case 0:
      CudaTransposeNaiveWrapped(d_mat, d_matT, rows, cols, 4, 1024);
      break;
    case 1:
      CudaTransposeIntermediateWrapped<32, 0>(d_mat, d_matT, rows, cols);
      break;
    case 2:
      CudaTransposeIntermediateWrapped<32, 1>(d_mat, d_matT, rows, cols);
      break;
    default:
      printf("Error unknown mode %d\n", mode);
      throw std::invalid_argument("Unknown mode");
  }
  cudaMemcpy(h_matT, d_matT, matrix_size, cudaMemcpyDeviceToHost);
  cudaFree(d_mat);
  cudaFree(d_matT);
}

void doTransposeCPU(float *h_mat, float *h_matT, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      h_matT[c * rows + r] = h_mat[r * cols + c];
    }
  }
}

void runTransposeExperiment(int matrix_rows, int matrix_cols, int mode) {
  float *h_mat = (float *)malloc(sizeof(float) * matrix_rows * matrix_cols);
  float *h_matT_experimental =
      (float *)malloc(sizeof(float) * matrix_rows * matrix_cols);
  float *h_matT_control =
      (float *)malloc(sizeof(float) * matrix_rows * matrix_cols);

  for (int i = 0; i < matrix_rows * matrix_cols; i++) {
    h_mat[i] = rand() % 100;
    h_matT_experimental[i] = 0;
    h_matT_control[i] = 0;
  }

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  doTransposeCUDA(h_mat, h_matT_experimental, matrix_rows, matrix_cols, mode);
  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CUDA CPU time %.5f (s)\n", cpu_time_used);

  start = clock();
  doTransposeCPU(h_mat, h_matT_control, matrix_rows, matrix_cols);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CPU time %.5f (s)\n", cpu_time_used);

  for (int i = 0; i < matrix_rows * matrix_cols; i++) {
    assert(fabs(h_matT_experimental[i] - h_matT_control[i]) < 0.001);
  }

  free(h_mat);
  free(h_matT_experimental);
  free(h_matT_control);
}

void runCopyThroughputExperiment(int matrix_rows, int matrix_cols) {
  int matrix_size_bytes = sizeof(float) * matrix_cols * matrix_rows;
  float *h_mat1 = (float *)malloc(sizeof(float) * matrix_size_bytes);
  float *h_mat2 = (float *)malloc(sizeof(float) * matrix_size_bytes);

  for (int i = 0; i < matrix_rows * matrix_cols; i++) {
    h_mat1[i] = rand() % 100;
    h_mat2[i] = 1;
  }

  float *d_mat1, *d_mat2;
  cudaMalloc((void **)&d_mat1, matrix_size_bytes);
  cudaMalloc((void **)&d_mat2, matrix_size_bytes);
  cudaMemcpy(d_mat1, h_mat1, matrix_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat2, h_mat2, matrix_size_bytes, cudaMemcpyHostToDevice);

  // Call kernel
  CudaBaselineCopyWrapped(d_mat1, d_mat2, matrix_rows, matrix_cols, 256);

  cudaMemcpy(h_mat2, d_mat2, matrix_size_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_mat1);
  cudaFree(d_mat2);

  for (int i = 0; i < matrix_rows * matrix_cols; i++) {
    assert(fabs(h_mat1[i] - h_mat2[i]) < 0.001);
  }
}

int main(int argc, char** argv) {
  int mode = 0;
  if (argc >= 2) {
    mode = atoi(argv[1]);
  }
  runCopyThroughputExperiment(1024, 1024 * 2);
  runTransposeExperiment(1024, 1024 * 2, mode);
  return 0;
}