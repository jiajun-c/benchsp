#include "cusp.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>

void cusparse_spmv(int *A_csrOffsets, int *A_columns, float *A_values, float *x, float* y, int row, int col, int nnz) {
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    cudaMalloc((void**)&dA_csrOffsets, (row+1)*sizeof(int));
    cudaMalloc((void**)&dA_columns, nnz*sizeof(int));
    cudaMalloc((void**)&dA_values, nnz*sizeof(float));
    cudaMalloc((void**)&dX, col*sizeof(float));
    cudaMalloc((void**)&dY, row*sizeof(float));

    cudaMemcpy(dA_csrOffsets, A_csrOffsets, (row + 1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, A_columns, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, A_values, nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, x, col*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, y, row*sizeof(float), cudaMemcpyHostToDevice);
    
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, row, col, nnz,
        dA_csrOffsets, dA_columns, dA_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, col, dX, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, row, dY, CUDA_R_32F);
    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, vecX, &beta, vecY,
                          CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
    float *dBuffer;
    cudaMalloc(&dBuffer, bufSize);
    cusparseSpMV_preprocess(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaMemcpy(y, dY, row * sizeof(float),cudaMemcpyDeviceToHost);
}