#include <cuda.h>
#include <stdio.h>
#include <time.h>

const int STRATEGY_ELEM = 0;
const int STRATEGY_ROW = 1;
const int STRATEGY_COL = 2;
const bool DEBUG = true;

__global__ void oneElementAdd(float *d_A, float *d_B, float *d_C, int n) {
  // exercise 3.1b: one element addition of matricies
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    d_C[i] = d_A[i] + d_B[i];
    if (DEBUG) printf("thread %d: index %d\n", i, i);
  }
}

__global__ void oneElementAddRow(float *d_A, float *d_B, float *d_C, int n,
                                 int row_size) {
  // exercise 3.1c: kernel that has each thread do one row
  int row_idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int ele_row = 0; ele_row < row_size; ele_row++) {
    int i = ele_row + row_idx * row_size;
    if (i < n) {
      d_C[i] = d_A[i] + d_B[i];
      if (DEBUG) printf("thread %d: index %d\n", row_idx, i);
    }
  }
}

__global__ void oneElementAddCol(float *d_A, float *d_B, float *d_C, int n,
                                 int col_size) {
  // exercise 3.1d: kernel that has each thread do one col
  int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int ele_col = 0; ele_col < col_size; ele_col++) {
    int i = col_size * ele_col + col_idx;
    if (i < n) {
      d_C[i] = d_A[i] + d_B[i];
      if (DEBUG) printf("thread %d: index %d\n", col_idx, i);
    }
  }
}

void print_matrix_flattened(float *matrix, int max_ele) {
  // prints first 10 elements
  printf("[");
  for (int i = 0; i < min(max_ele, 10); i++) {
    printf(" %.1f ", matrix[i]);
  }
  printf(" ...]");
}

double run_code(int matrix_rows, int matrix_cols, int threads_per_block,
                int mode) {
  // 3.1a - harness for host code
  int matrix_element_count = matrix_rows * matrix_cols;
  size_t matrix_size_bytes = sizeof(float) * matrix_element_count;

  // host data
  float *h_A = (float *)malloc(matrix_size_bytes);
  float *h_B = (float *)malloc(matrix_size_bytes);
  float *h_C = (float *)malloc(matrix_size_bytes);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    return 1; // failed :(
  }

  for (int i = 0; i < min(10, matrix_element_count); i++) {
    h_A[i] = rand() % 10;
    h_B[i] = rand() % 10;
  }

  // device data
  float *d_A, *d_B, *d_C;

  cudaMalloc((void **)&d_A, matrix_size_bytes);
  cudaMalloc((void **)&d_B, matrix_size_bytes);
  cudaMalloc((void **)&d_C, matrix_size_bytes);

  cudaMemcpy(d_A, h_A, matrix_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, matrix_size_bytes, cudaMemcpyHostToDevice);

  // TODO: launch kernel and copy results
  clock_t start, end;
  double cpu_time_used;

  switch (mode) {
  case STRATEGY_ELEM:
    printf("Using mode elemwise CUDA\n");
    start = clock();
    oneElementAdd<<<ceil(matrix_element_count / (float)threads_per_block),
                    threads_per_block>>>(d_A, d_B, d_C, matrix_element_count);
    end = clock();
    break;
  case STRATEGY_ROW:
    printf("Using mode row CUDA\n");
    start = clock();
    oneElementAddRow<<<ceil(matrix_rows / (float)threads_per_block),
                       threads_per_block>>>(d_A, d_B, d_C, matrix_element_count, matrix_cols);
    end = clock();
    break;
  case STRATEGY_COL:
    printf("Using mode col CUDA\n");
    start = clock();
    oneElementAddCol<<<ceil(matrix_cols / (float)threads_per_block),
                       threads_per_block>>>(d_A, d_B, d_C, matrix_element_count, matrix_rows);
    end = clock();
    break;
  default:
    printf("Unknown mode %d", mode);
  }
  cudaDeviceSynchronize();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  cudaMemcpy(h_C, d_C, matrix_size_bytes, cudaMemcpyDeviceToHost);

  print_matrix_flattened(h_A, matrix_element_count);
  printf(" + ");
  print_matrix_flattened(h_B, matrix_element_count);
  printf(" = ");
  print_matrix_flattened(h_C, matrix_element_count);
  printf("\nTime Elapsed : %.10f (s)\n\n", cpu_time_used);

  // Free resources
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  return cpu_time_used;
}

int main() {
  int MATRIX_ROWS = 10000;
  int MATRIX_COLS = 10000;
  int THREAD_SIZE = 8;

  printf("**************\n");
  run_code(MATRIX_ROWS, MATRIX_COLS, THREAD_SIZE, STRATEGY_ELEM);
  run_code(MATRIX_ROWS, MATRIX_COLS, THREAD_SIZE, STRATEGY_ROW);
  run_code(MATRIX_ROWS, MATRIX_COLS, THREAD_SIZE, STRATEGY_COL);

  // exercise 3.1e:  
  // bigger row sizes hurts strategy where one thread processes row
  printf("**************\n");
  run_code(100, 10000, THREAD_SIZE, STRATEGY_ELEM);
  run_code(100, 10000, THREAD_SIZE, STRATEGY_ROW); // a bit slower
  run_code(100, 10000, THREAD_SIZE, STRATEGY_COL);

  // bigger col sizes hurts strategy where one thread processes col
  printf("**************\n");
  run_code(10000, 100, THREAD_SIZE, STRATEGY_ELEM);
  run_code(10000, 100, THREAD_SIZE, STRATEGY_ROW); 
  run_code(10000, 100, THREAD_SIZE, STRATEGY_COL); // not a bit slower!
  // might need to bump up array sizes?

  // on the other hand there are probably happy mediums for col and rows where
  // each strategy will shine.
  // While the cost of a thread in CUDA is low, there is still overhead so
  // we might need to process more than one element to hit each compute unit's capacity.
  // furthermore there might be caching effects. The ideal workload for the row compute
  // for example is one where there are enough rows to create enough parallel threads
  // to saturate the compute capabilites of the device but no more rows than this to
  // avoid extraneous threads.
  return 0;
}