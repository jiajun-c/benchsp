#include "cusp.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
int cusparse_spmv(int *hA_csrOffsets, int64_t *hA_columns, float *hA_values, float *hX, float* hY, int64_t A_num_rows, int64_t A_num_cols, int64_t A_nnz) {
    // float     alpha           = 1.0f;
    // float     beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    // const int A_num_rows      = 4;
    // const int A_num_cols      = 4;
    // const int A_nnz           = 9;
    // int       hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    // int       hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    // float     hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                               6.0f, 7.0f, 8.0f, 9.0f };
    // float     hX[]            = { 1.0f, 1.0f, 1.0f, 1.0f };
    // float     hY[]            = { 0.0f, 0.0f, 0.0f, 0.0f };
    // float     hY_result[]     = { 19.0f, 8.0f, 51.0f, 52.0f };
    // hY[0] = 3.0;
    // printf("A_num_rows %d %d %d\n", A_num_rows, A_num_cols, A_nnz);
    // for (int i = 0; i < 5; i++) {
    //     printf("A_csrOffsets %d %d\n", i, hA_csrOffsets[i]);

    // }
    // for (int i = 0; i < A_nnz; i++) {
    //     printf("hA_columns %d %d\n", i, hA_columns[i]);

    // }
    //     for (int i = 0; i < A_nnz; i++) {
    //     printf("hA_values %d %f\n", i, hA_values[i]);

    // }
    //     for (int i = 0; i < 4; i++) {
    //     printf("hX %d %f\n", i, hX[i]);

    // }
    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    // CHECK_CUSPARSE( cusparseSpMV_preprocess(
    //                              handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                              &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
    //                              CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),cudaMemcpyDeviceToHost) )
    // hY[0] = 3.0;
    // for (int i = 0; i < A_num_rows; i++)
    // printf("%f\n", hY[i]);
}