#include <cuda.h> 
#include <stdio.h>
#include <assert.h>

__global__ 
void MatrixVectorMultCuda(float* B, float* C, float* A, int rows_B, int cols_B) {
    // i indexes into the result of A
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < rows_B) { 
        A[i] = 0;
        for (int reduction_axis = 0; reduction_axis < cols_B; reduction_axis++) {
            int index_b = i * cols_B + reduction_axis;
            A[i] += B[index_b] * C[reduction_axis];
        }
    }
}

void do_matrix_vector_multiplication_cuda(float* h_B, float* h_C, float* h_A, int rows_B, int cols_B, int threads_per_block) {
    // A[i] = sum_j B[i][j] * C[j]

    int B_ele_cnt = rows_B * cols_B;
    size_t B_size = B_ele_cnt * sizeof(float);
    size_t C_size = cols_B * sizeof(float);
    size_t A_size = rows_B * sizeof(float);
    float* d_B, *d_C, *d_A;

    cudaMalloc((void**)&d_B, B_size);
    cudaMalloc((void**)&d_C, C_size);
    cudaMalloc((void**)&d_A, A_size);
    cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, C_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);

    // Call kernel
    MatrixVectorMultCuda<<<ceil(rows_B / (float)threads_per_block), threads_per_block>>>(d_B, d_C, d_A, rows_B, cols_B);

    cudaMemcpy(h_A, d_A, A_size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void do_matrix_vector_multiplication_cpu(float* h_B, float* h_C, float* h_A, int rows_B, int cols_B) {
    for (int row_i = 0; row_i < rows_B; row_i++) {
        h_A[row_i] = 0;
        for (int col_i = 0; col_i < cols_B; col_i++) {
            int index_b = row_i * cols_B + col_i;
            h_A[row_i] += h_B[index_b] * h_C[col_i];
        }
    }
}

int main() {
  int MATRIX_ROWS = 10000;
  int MATRIX_COLS = 10000;
  int THREAD_SIZE = 256;

  float* h_B = (float*)malloc(sizeof(float) * MATRIX_ROWS * MATRIX_COLS);
  float* h_C =  (float*)malloc(sizeof(float) * MATRIX_COLS);
  float* h_A1 =  (float*)malloc(sizeof(float) * MATRIX_ROWS);
  float* h_A2 =  (float*)malloc(sizeof(float) * MATRIX_ROWS);

  for (int i = 0; i < MATRIX_ROWS * MATRIX_COLS; i++) {
      h_B[i] = rand() % 100;
  }
  for (int i = 0; i < MATRIX_COLS; i++) {
      h_C[i] = rand() % 100;
  }

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  do_matrix_vector_multiplication_cuda(h_B, h_C, h_A1, MATRIX_ROWS, MATRIX_COLS, THREAD_SIZE);
  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CUDA time %.5f (s)\n", cpu_time_used);

  start = clock();
  do_matrix_vector_multiplication_cpu(h_B, h_C, h_A2, MATRIX_ROWS, MATRIX_COLS);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CPU time %.5f (s)\n", cpu_time_used);

  for (int i = 0; i < MATRIX_ROWS; i++) {
      assert(fabs(h_A1[i] - h_A2[i]) < 0.001);
  }

  free(h_B);
  free(h_C);
  free(h_A1);
  free(h_A2);
  return 0;
}