#include <iostream>
void ref_spmv(int *A_csrOffsets, int64_t *A_columns, float *A_values, float *x, float* y, int64_t row, int64_t col);
void ref_spmv_fp64(int *A_csrOffsets, int64_t *A_columns, double *A_values, double *x, double* y, int64_t row, int64_t col);