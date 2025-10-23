#include <iostream>
#include "matrixFormat.hpp"
int amgT_spmv_fp64(int32_t *hA_csrOffsets, int32_t *hA_columns, double *hA_values, double *hX, double* hY, int32_t A_num_rows, int32_t A_num_cols, int32_t A_nnz, int repeat,  double& avg_time);
int amgT_spmv_fp64_dbsr(dbsrMat *dbsr, double *hX, double *hY,int32_t warmup, int32_t repeat, double& avg_time);
// two level balance
int amgT_spmv_fp64_dbsr_balance(dbsrMat *dbsr, double *hX, double *hY,int32_t warmup, int32_t repeat, double &avg_time);