#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define N 4096

extern "C" {
extern const unsigned long long fatbinData[];
}

extern "C" __global__ void saxpy(float a, float *x, float *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  y[i] = a * x[i] + y[i];
}

int main(void) {
  float *x, *y, *d_x, *d_y;

  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  cuInit(0);

  CUdevice pdev;
  cuDeviceGet(&pdev, 0);

  CUcontext pctx;
  cuCtxCreate(&pctx, 0, pdev);

  CUmodule mod = 0;
  cuModuleLoadData(&mod, fatbinData);
  cuModuleLoadFatBinary(&mod, fatbinData);

  CUfunction f = 0;
  cuModuleGetFunction(&f, mod, "saxpy");
  assert(f != 0);

  cuMemAlloc((CUdeviceptr *)&d_x, N * sizeof(float));
  cuMemAlloc((CUdeviceptr *)&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cuMemcpy((CUdeviceptr)d_x, (CUdeviceptr)x, N * sizeof(float));
  cuMemcpy((CUdeviceptr)d_y, (CUdeviceptr)y, N * sizeof(float));

  float a = 2.0f;
  void *args[] = {&a, &d_x, &d_y};

  cuLaunchKernel(f, N / 256, 1, 1, 256, 1, 1, 0, 0, args, NULL);

  return 0;
}

