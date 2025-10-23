#include <iostream>
void ref_spmv_fp16(int *A_csrOffsets, int32_t *A_columns, float *A_values, float *x, float* y, int32_t row, int32_t col);
void ref_spmv_fp32(int *A_csrOffsets, int32_t *A_columns, float *A_values, float *x, float* y, int32_t row, int32_t col);
void ref_spmv_fp64(int64_t *A_csrOffsets, int64_t *A_columns, double *A_values, double *x, double* y, int64_t row, int64_t col, double& avg_time);