#include "ref.h"
#include <cuda_fp16.h>
void ref_spmv_fp16(int *A_csrOffsets, int32_t *A_columns, float *A_values, float *x, float* y, int32_t row, int32_t col) {
    int base = 0;
    for (int i = 0; i < row; i++) {
        float sum = 0;
        for (int j = A_csrOffsets[i]; j < A_csrOffsets[i+1]; j++) {
            sum +=__half2float(A_values[j])  * __half2float(x[A_columns[j]]);
        }
        // base = A_csrOffsets[i];
        y[i] = half(sum);
    }
}
void ref_spmv_fp32(int *A_csrOffsets, int32_t *A_columns, float *A_values, float *x, float* y, int32_t row, int32_t col) {
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

void ref_spmv_fp64(int64_t *A_csrOffsets, int64_t *A_columns, double *A_values, double *x, double* y, int64_t row, int64_t col, double& avg_time) {
    int base = 0;
    
    for (int64_t i = 0; i < row; i++) {
        double sum = 0;
        for (int64_t j = A_csrOffsets[i]; j < A_csrOffsets[i+1]; j++) {
            sum += A_values[j] * x[A_columns[j]];
        }
        // base = A_csrOffsets[i];
        y[i] = sum;
    }
}
