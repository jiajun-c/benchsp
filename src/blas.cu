#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
using namespace std;

int gemvfp64(double *A, double *B, double *C, int M, int N, int K, int repeat) {
    cublasStatus_t status;
    double alpha = 1.0;
    double beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * N * sizeof(double));
    cudaMalloc((void **)&d_B, N * sizeof(double));
    cudaMalloc((void **)&d_C, M * sizeof(double));
    cudaMemcpy(d_A, A, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * sizeof(double), cudaMemcpyHostToDevice);
    for (int i = 0; i < repeat; i++) {
        auto start = chrono::high_resolution_clock::now();
        status = cublasDgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_B, 1, &beta, d_C, 1);
        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "dense blas 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    }
    cudaMemcpy(C, d_C, M * sizeof(double), cudaMemcpyDeviceToHost);
    return status;
}