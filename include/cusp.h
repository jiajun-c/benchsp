#include <iostream>
#include <cuda_fp16.h>
int cusparse_spmv(int *A_csrOffsets, int64_t *A_columns, float *A_values, float *x, float* y, int64_t row, int64_t col, int64_t nnz);
int cusparse_spmv_fp64(int64_t *hA_csrOffsets, int64_t *hA_columns, double *hA_values, double *hX, double* hY, int64_t A_num_rows, int64_t A_num_cols, int64_t A_nnz, int repeat);
int cusparse_spmv_fp16(int *hA_csrOffsets, int32_t *hA_columns, half *hA_values, half *hX, half* hY, int32_t A_num_rows, int32_t A_num_cols, int32_t A_nnz);
void cusparse_spmv_fp16_warpper(int *hA_csrOffsets, int32_t *hA_columns, float *hA_values32, float *hX32, float* hY32, int32_t A_num_rows, int32_t A_num_cols, int32_t A_nnz);