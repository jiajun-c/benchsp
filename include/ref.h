#include <iostream>
void ref_spmv(int64_t *A_csrOffsets, int64_t *A_columns, float *A_values, float *x, float* y, int64_t row, int64_t col);