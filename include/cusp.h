#include <iostream>
int cusparse_spmv(int *A_csrOffsets, int64_t *A_columns, float *A_values, float *x, float* y, int64_t row, int64_t col, int64_t nnz);
int cusparse_spmv_fp64(int *hA_csrOffsets, int64_t *hA_columns, double *hA_values, double *hX, double* hY, int64_t A_num_rows, int64_t A_num_cols, int64_t A_nnz);
