/*
Playing around with implementing layout transform / transposition efficiently 
*/
#include <cuda.h> 
#include <stdio.h>
#include <assert.h>

__global__ 
void CudaTransposeNaive(
    float* mat, 
    float* matT, 
    int rows, 
    int cols, 
    int loads_per_thread,
    int num_threads_per_block
) {
    int y_start = blockIdx.y * loads_per_thread; 
    int x_start = blockIdx.x * num_threads_per_block + threadIdx.x;
    for (int i = 0; i < loads_per_thread; i++) { 
        matT[x_start * rows + (y_start + i)] = mat[(y_start + i) * cols + x_start];
    }
}

void CudaTransposeNaiveWrapped(float* d_mat, float* d_matT, int rows, int cols, int loads_per_thread=32, int num_threads_per_block=32) {
    int num_blocks_y = rows / loads_per_thread;
    assert(rows % loads_per_thread == 0);
    int num_blocks_x = cols / num_threads_per_block;
    assert(cols % num_threads_per_block == 0);
    dim3 dimBlock(num_blocks_x, num_blocks_y, 1);
    dim3 dimGrid(loads_per_thread, 1, 1);
    CudaTransposeNaive<<<dimBlock, dimGrid>>>(d_mat, d_matT, rows, cols, loads_per_thread, num_threads_per_block);
}

void doTransposeCUDA(float* h_mat, float* h_matT, int rows, int cols) {

    int matrix_size = rows * cols * sizeof(float);
    float* d_mat, *d_matT;

    cudaMalloc((void**)&d_mat, matrix_size);
    cudaMalloc((void**)&d_matT, matrix_size);
    cudaMemcpy(d_mat, h_mat, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matT, h_matT, matrix_size, cudaMemcpyHostToDevice);

    // Call kernel
    CudaTransposeNaiveWrapped(d_mat, d_matT, rows, cols, 4, 1024);

    cudaMemcpy(h_matT, d_matT, matrix_size, cudaMemcpyDeviceToHost);
    cudaFree(d_mat);
    cudaFree(d_matT);
}

void doTransposeCPU(float* h_mat, float* h_matT, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            h_matT[c * rows + r] = h_mat[r * cols + c];
        }
    }
}

int main() {
  int MATRIX_ROWS = 1024 * 8;
  int MATRIX_COLS = 1024 * 8;

  float* h_mat = (float*)malloc(sizeof(float) * MATRIX_ROWS * MATRIX_COLS);
  float* h_matT_experimental =  (float*)malloc(sizeof(float) * MATRIX_ROWS * MATRIX_COLS);
  float* h_matT_control =  (float*)malloc(sizeof(float) * MATRIX_ROWS * MATRIX_COLS);

  for (int i = 0; i < MATRIX_ROWS * MATRIX_COLS; i++) {
      h_mat[i] = rand() % 100;
      h_matT_experimental[i] = 0;
      h_matT_control[i] = 0;
  }

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  doTransposeCUDA(h_mat, h_matT_experimental, MATRIX_ROWS, MATRIX_COLS);
  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CUDA time %.5f (s)\n", cpu_time_used);

  start = clock();
  doTransposeCPU(h_mat, h_matT_control, MATRIX_ROWS, MATRIX_COLS);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CPU time %.5f (s)\n", cpu_time_used);

  for (int i = 0; i < MATRIX_ROWS; i++) {
      assert(fabs(h_matT_experimental[i] - h_matT_control[i]) < 0.001);
  }

  free(h_mat);
  free(h_matT_experimental);
  free(h_matT_control);
  return 0;
}