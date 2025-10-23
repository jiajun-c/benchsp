#include <iostream>
int spmv_fp64_warpper(int32_t *rowPtr, int32_t *colIdx, int32_t rows, int32_t cols, int32_t nnz, double* values,double* x, double* outvalues, int repeat, double& avg_time);
int spmv_fp64_balance_warpper(int32_t *rowPtr, int32_t *colIdx, int32_t rows, int32_t cols, int32_t nnz, double* values,double* x, double* outvalues, int repeat, double& avg_time);