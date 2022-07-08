#include <cuda.h>
#include <iomanip>
#include <iostream>
// Tiled matrix multiplication, does not implement edge conditions however

// Tolerance for comparing floating points
#define EPSILON 0.0001

// Size of tiles
#define TILE_SIZE 32

#define DEBUG false 

const int MODE_CPU = 0;
const int MODE_GPU_NAIVE = 1;
const int MODE_GPU_TILED = 2;

__global__ void matrix_multiplication_tiled_cuda(float *A, float *B, float *C,
                                                 int dimension_size) {
  int tile_col = threadIdx.x;
  int tile_row = threadIdx.y;

  int output_col = blockIdx.x * blockDim.x + tile_col;
  int output_row = blockIdx.y * blockDim.y + tile_row;

  // shared memory
  __shared__ float A_s[TILE_SIZE][TILE_SIZE];
  __shared__ float B_s[TILE_SIZE][TILE_SIZE];

  // NOTE: This does not handle edge conditions
  for (int phase_number = 0; phase_number < dimension_size / TILE_SIZE;
       phase_number++) {
    // load stuff into shared memory
    int col_into_A = TILE_SIZE * phase_number + tile_col;
    int linearized_A_i = output_row * dimension_size + col_into_A;
    A_s[tile_row][tile_col] = A[linearized_A_i];

    int row_into_B = TILE_SIZE * phase_number + tile_row;
    int linearized_B_i = row_into_B * dimension_size + output_col;
    B_s[tile_row][tile_col] = B[linearized_B_i];

    __syncthreads();

    float local_result = 0;

    // result of partial matrix multiplication into ele
    for (int reduction_axis = 0; reduction_axis < TILE_SIZE; reduction_axis++) {
      if (DEBUG) {
      printf("Thread (%d, %d) in block (%d, %d) in phase %d "
             "has A_s = %f and B_s = %f with linearized_A_i = %d and lineared_B_i = %d\n",
             tile_col, tile_row, blockIdx.x, blockIdx.y, phase_number,
             A_s[tile_row][reduction_axis], B_s[reduction_axis][tile_col],
             linearized_A_i, linearized_B_i);
      }
      local_result +=
          A_s[tile_row][reduction_axis] * B_s[reduction_axis][tile_col];
    }

    C[output_row * dimension_size + output_col] += local_result;
    __syncthreads();
  }
}

__global__ void matrix_multiplication_naive_cuda(float *A, float *B, float *C,
                                                 int dimension_size) {
  // naive, probably slow algo
  int output_row = blockIdx.x * blockDim.x + threadIdx.x;
  int output_col = blockIdx.y * blockDim.y + threadIdx.y;

  if (output_row < dimension_size && output_col < dimension_size) {
    float result = 0;
    for (int reduction_axis = 0; reduction_axis < dimension_size;
         reduction_axis++) {
      result += A[output_row * dimension_size + reduction_axis] *
                B[reduction_axis * dimension_size + output_col];
    }

    int flattened_idx = output_row * dimension_size + output_col;
    C[flattened_idx] = result;
  }
}

void matrix_multiplication_naive_cpu(float *A, float *B, float *C,
                                     int dimension_size) {
  for (int i = 0; i < dimension_size; i++) {
    for (int j = 0; j < dimension_size; j++) {
      float result = 0;
      for (int k = 0; k < dimension_size; k++) {
        int i_A = i * dimension_size + k;
        int i_B = k * dimension_size + j;
        result += A[i_A] * B[i_B];
      }
      int i_C = i * dimension_size + j;
      C[i_C] = result;
    }
  }
}

void initialize_matrix(float *res, int dimension_size) {
  for (int i = 0; i < dimension_size * dimension_size; i++) {
    res[i] = rand() % 10;
  }
}

void print_matrix(float *matrix, int dimension_size, int spacing = 5) {
  // print dimension_size x dimension_size matrix
  for (int i = 0; i < dimension_size * spacing; i++) {
    std::cout << "-";
  }
  std::cout << std::endl;

  for (int i = 0; i < dimension_size; i++) {
    for (int j = 0; j < dimension_size; j++) {
      std::cout << std::setw(spacing) << matrix[i * dimension_size + j];
    }
    std::cout << std::endl;
  }

  for (int i = 0; i < dimension_size * spacing; i++) {
    std::cout << "-";
  }
  std::cout << std::endl;
}

bool are_same(float *A, float *B, int dimension_size) {
  for (int i = 0; i < dimension_size * dimension_size; i++) {
    if (fabs(A[i] - B[i]) > EPSILON)
      return false;
  }
  return true;
}

std::pair<float *, double> run_code(float *h_A, float *h_B, int mode, int N) {
  // returns output matrix and double
  float *h_C = (float *)calloc(N * N, sizeof(float));
  float *d_A, *d_B, *d_C;

  size_t matrix_size_bytes = N * N * sizeof(float);
  if (mode != MODE_CPU) {
    // move things to GPU
    cudaMalloc((void **)&d_A, matrix_size_bytes);
    cudaMalloc((void **)&d_B, matrix_size_bytes);
    cudaMalloc((void **)&d_C, matrix_size_bytes);
    cudaMemcpy(d_A, h_A, matrix_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size_bytes, cudaMemcpyHostToDevice);
  }

  clock_t start, end;
  double cpu_time_used;
  start = clock();
  end = clock();

  // keep to square dims for now...
  int grid_dim = N / TILE_SIZE; // TODO: handle edge conditions
  dim3 dimGrid(grid_dim, grid_dim);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  switch (mode) {
  case MODE_CPU:
    start = clock();
    matrix_multiplication_naive_cpu(h_A, h_B, h_C, N);
    end = clock();
    break;
  case MODE_GPU_NAIVE:
    start = clock();
    matrix_multiplication_naive_cuda<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    end = clock();
    break;
  case MODE_GPU_TILED:
    start = clock();
    matrix_multiplication_tiled_cuda<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    end = clock();
    break;
  default:
    std::cout << "Unknown mode " << mode << std::endl;
  }
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

  if (mode != MODE_CPU) {
    cudaMemcpy(h_C, d_C, matrix_size_bytes, cudaMemcpyDeviceToHost);
  }

  if (mode != MODE_CPU) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }

  return {h_C, cpu_time_used};
}

int main() {
  // Multiplying N x N and N x N matrix
  int N = 1024; // make multiple of TILE_SIZE
  float *h_A = (float *)malloc(sizeof(float) * N * N);
  float *h_B = (float *)malloc(sizeof(float) * N * N);

  initialize_matrix(h_A, N);
  initialize_matrix(h_B, N);

  // check correctness
  std::pair<float *, double> result_cpu = run_code(h_A, h_B, MODE_CPU, N);
  std::pair<float *, double> result_gpu_naive =
      run_code(h_A, h_B, MODE_GPU_NAIVE, N);
  std::pair<float *, double> result_gpu_tiled =
      run_code(h_A, h_B, MODE_GPU_TILED, N);

    
  std::cout << std::fixed << std::setprecision(8) << result_cpu.second << " s on CPU\n";
  std::cout << std::fixed << std::setprecision(8) << result_gpu_naive.second << " s on GPU naive\n";
  std::cout << std::fixed << std::setprecision(8) << result_gpu_tiled.second << " s on GPU tiled\n";

  std::cout << "Do CPU and GPU naive match: "
            << are_same(result_cpu.first, result_gpu_naive.first, N)
            << std::endl;
  std::cout << "Do CPU and GPU tiled match: "
            << are_same(result_cpu.first, result_gpu_tiled.first, N)
            << std::endl;

  if (DEBUG) {
    std::cout << "A:\n";
    print_matrix(h_A, N);
    std::cout <<"B:\n";
    print_matrix(h_B, N);
    std::cout << "GPU\n";
    print_matrix(result_gpu_tiled.first, N);
    std::cout << "CPU\n";
    print_matrix(result_cpu.first, N);
  }

  free(h_A);
  free(h_B);
  free(result_cpu.first);
  free(result_gpu_naive.first);
  free(result_gpu_tiled.first);
  return 0;
}