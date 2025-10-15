#include <iostream>
#include <cuda_fp16.h>
void bcsr_spmv_fp64(int64_t *hA_csrOffsets, int64_t *hA_columns, double *hA_values32, double *hX32, double* hY32, int64_t A_num_rows, int64_t A_num_cols, int64_t A_nnz, int repeat);