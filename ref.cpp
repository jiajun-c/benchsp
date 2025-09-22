#include "ref.h"

void ref_spmv(int *A_csrOffsets, int64_t *A_columns, float *A_values, float *x, float* y, int64_t row, int64_t col) {
    int base = 0;
    for (int i = 0; i < row; i++) {
        float sum = 0;
        for (int j = A_csrOffsets[i]; j < A_csrOffsets[i+1]; j++) {
            sum += A_values[j] * x[A_columns[j]];
        }
        // base = A_csrOffsets[i];
        y[i] = sum;
    }
}

void ref_spmv_fp64(int *A_csrOffsets, int64_t *A_columns, double *A_values, double *x, double* y, int64_t row, int64_t col) {
    int base = 0;
    for (int i = 0; i < row; i++) {
        double sum = 0;
        for (int j = A_csrOffsets[i]; j < A_csrOffsets[i+1]; j++) {
            sum += A_values[j] * x[A_columns[j]];
        }
        // base = A_csrOffsets[i];
        y[i] = sum;
    }
}