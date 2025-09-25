#include <iostream>
int cusparse_spmv(int *A_csrOffsets, int64_t *A_columns, float *A_values, float *x, float* y, int64_t row, int64_t col, int64_t nnz);
int cusparse_spmv_fp64(int *hA_csrOffsets, int64_t *hA_columns, double *hA_values, double *hX, double* hY, int64_t A_num_rows, int64_t A_num_cols, int64_t A_nnz);
int cusparse_spmm_fp16(int *hA_csrOffsets, int32_t *hA_columns, float *hA_values32,float *hB32, float* hC32, int M, int N, int K, int A_nnz);