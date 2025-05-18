#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define N 4096

__global__ void gemm(const float *A, const float *B, float *C) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if(i < N && j < N){
    float inner_prod = 0.0f;
    for(int k = 0; k < N; k++){
      inner_prod += A[i*N+k] * B[k*N+j];
    }
    C[i*N+j] = inner_prod;
  }
}

void check(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

int main(void) {
  size_t size = N * N * sizeof(float);

  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);

  for (int i = 0; i < N * N; i++) h_A[i] = 1.0f;
  for (int i = 0; i < N * N; i++) h_B[i] = 1.0f;

  float *d_A, *d_B, *d_C;
  check(cudaMalloc(&d_A, size));
  check(cudaMalloc(&d_B, size));
  check(cudaMalloc(&d_C, size));

  check(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

  gemm<<<blocks, threads>>>(d_A, d_B, d_C);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());

  check(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  // Quick check (optional)
  float expected = N * 1.0f;
  bool ok = true;
  for (int i = 0; i < N * N; i++) {
    if (fabs(h_C[i] - expected) > 1e-3) {
      ok = false;
      printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
      break;
    }
  }

  printf(ok ? "PASS\n" : "FAIL\n");

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C);
  return 0;
}

