#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>
using namespace std;


// #define WARP_SIZE 32
// #define BlockSize 8

// #define MMA_M 8
// #define MMA_N 8
// #define MMA_K 4
// #define groupNum 1
// #define warpNum_short 4
// #define loopNum_short 4
// #define warpNum_long 4
// #define loopNum_long 2

// __device__ __forceinline__ half warpReduceSum(half sum){
//     sum += __shfl_down_sync(0xffffffff, sum, 16);
//     sum += __shfl_down_sync(0xffffffff, sum, 8);
//     sum += __shfl_down_sync(0xffffffff, sum, 4);
//     sum += __shfl_down_sync(0xffffffff, sum, 2);
//     sum += __shfl_down_sync(0xffffffff, sum, 1);
//     return sum;
// }
// __device__ __forceinline__ void mma_m8n8k4_fp16(half *acc, half *frag_a, half *frag_b)
// {
//     uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_a[0]);
//     uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
//     uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

//     asm volatile(
//         "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
//         " { %0, %1, %2, %3 }, "
//         " { %4, %5 }, "
//         " { %6, %7 }, "
//         " { %0, %1, %2, %3 };"
//         : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
//         "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
//     ); 
// }

// __device__ __forceinline__ void mma_m8n8k4_fp16_v2(half *acc, uint32_t *A, half *frag_b)
// {
//     uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
//     uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

//     asm volatile(
//         "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
//         " { %0, %1, %2, %3 }, "
//         " { %4, %5 }, "
//         " { %6, %7 }, "
//         " { %0, %1, %2, %3 };"
//         : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
//         "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
//     ); 
// }

// __device__ __forceinline__ void mma_m8n8k4_fp16_v3(uint32_t *C, uint32_t *A, uint32_t *B)
// {
//     asm volatile(
//         "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
//         " { %0, %1, %2, %3 }, "
//         " { %4, %5 }, "
//         " { %6, %7 }, "
//         " { %0, %1, %2, %3 };"
//         : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
//         "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
//     ); 
// }

// __global__ void dasp_spmv(uint32_t *dX_val) {
//     int bid = blockIdx.x;
//     int tid = threadIdx.x;
//     int laneID = 31 & tid;
//     int row = laneid < 16 ? (laneid >> 2) * 8 + (3 & laneid) : ((laneid - 16) >> 2) * 8 + (3 & laneid) + 4;
//     int idx = row * MMA_K;
//     int idx_val = row * 2;
//     int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;
//     half const *valX_half = reinterpret_cast<half const *>(&dX_val[0]);
//     half *valY_half = reinterpret_cast<half *>(&dY_val[0]);

// }

// void dasp_spmv_fp16(half *csrValA, int* csrRowPtrA, int*csrColIdxA, half* X_val, int nnz, int rows, int cols) {

//     int row_long = 0, row_block = 0, row_zero = 0;
//     int short_row_1 = 0, short_row_3 = 0, short_row_2 = 0, short_row_4 = 0;
//     for (int i = 0; i < rows; i ++)
//     {
//         int row_len = csrRowPtrA[i + 1] - csrRowPtrA[i];
//         if (row_len == 1)
//         {   
//             short_row_1 ++;
//         }
//         else if (row_len == 3)
//         {
//             short_row_3 ++;
//         }
//         else if (row_len == 2)
//         {
//             short_row_2 ++;
//         }
//         else if (row_len == 0)
//         {
//             row_zero ++;
//         }
//         else if (row_len == 4)
//         {
//             short_row_4 ++;
//         }
//         // else if (row_len >= warpNum_long * loopNum_long * MMA_M * MMA_K)
//         else if (row_len >= block_longest)
//         {
//             row_long ++;
//         }
//         else
//         {
//             row_block ++;
//         }
//     }
//     // 可能是循环展开？
//     int rowloop;
//     if (row_block < 59990) rowloop = 1;
//     else if (row_block >= 59990 && row_block < 400000) rowloop = 2;
//     else rowloop = 4;

//     // 表示各类长短行对应的实际行id
//     int *short_rid_1 = (int *)malloc(sizeof(int) * short_row_1);
//     int *short_rid_2 = (int *)malloc(sizeof(int) * short_row_2);
//     int *short_rid_3 = (int *)malloc(sizeof(int) * short_row_3);
//     int *short_rid_4 = (int *)malloc(sizeof(int) * short_row_4);
//     int *long_rid = (int *)malloc(sizeof(int) * row_long);
//     int *zero_rid = (int *)malloc(sizeof(int) * row_zero);
//     int *ridA = (int *)malloc(sizeof(int) * row_block);
//     int *rptA = (int *)malloc(sizeof(int) * row_block + 1);
//     memset(rptA, 0, sizeof(int) * row_block + 1);

//     int long_rpt = (int *)malloc(sizeof(int) * row_long + 1);
//     memset(long_rpt, 0, sizeof(int) * row_long + 1);

//     int short_row_flag1 = 0, short_row_flag3 = 0, short_row_flag2 = 0, short_row_flag4 = 0;
//     int row_long_flag = 0, flag0 = 0, row_block_flag = 0;

//     for (int i = 0; i < rows; i++) {
//         int row_len = csrRowPtrA[i + 1] - csrRowPtrA[i];
//         if (row_len == 1) {
//             short_rid_1[short_row_flag1] = i;
//             short_row_flag1++;
//         } else if (row_len == 2) {
//             short_rid_2[short_row_flag2] = i;
//             short_row_flag2++;
//         } else if (row_len == 3) {
//             short_rid_3[short_row_flag3] = i;
//             short_row_flag3++;
//         } else if (row_len == 0)
//         {
//             zero_rid[flag0] = i;
//             flag0 ++;
//         }
//         else if (row_len == 4)
//         {
//             short_rid_4[short_row_flag4] = i;
//             short_row_flag4 ++;
//         }
//         // else if (row_len >= warpNum_long * loopNum_long * MMA_M * MMA_K)
//         else if (row_len >= block_longest)
//         {
//             long_rpt[row_long_flag] = row_len;
//             long_rid[row_long_flag] = i;
//             row_long_flag ++;
//         }
//         else
//         {
//             rptA[row_block_flag] = row_len;
//             ridA[row_block_flag] = i;
//             row_block_flag ++;
//         }
//     }

//     int nnz_short = short_row_1 + short_row_2*2 + short_row_3*3 + short_row_4*4;

//     // 表示行内只有1/3个元素的最小行数
//     int common_13 = short_row_1 < short_row_3 ? short_row_1 : short_row_3;
//     // 块大小为8
//     if (common_13 / BlockSize >= 16)
//     {
//         common_13 = BlockSize * 4 * (common_13 / (BlockSize * 4));
//         short_row_1 = short_row_1 - common_13;
//         short_row_3 = short_row_3 - common_13;
//     }
//     else
//     {
//         common_13 = 0;
//     }

//     int short_block13 = (common_13 + BlockSize - 1) / BlockSize;  
//     int half_short_row_2 = (short_row_2 + 1) / 2;
//     int short_block22 = (half_short_row_2 + BlockSize - 1) / BlockSize;
//     int short_row_34 = short_row_3 + short_row_4;
//     int short_block34 = (short_row_34 + BlockSize - 1) / BlockSize;
//     int block13_per_threadblock = warpNum_short * groupNum * 4;
//     int block22_per_threadblock = warpNum_short * groupNum * 4;
//     int block34_per_threadblock = warpNum_short * groupNum * loopNum_short;

//     int threadblock13 = (short_block13 + block13_per_threadblock - 1) / block13_per_threadblock;
//     int threadblock22 = (short_block22 + block22_per_threadblock - 1) / block22_per_threadblock;
//     int threadblock34 = (short_block34 + block34_per_threadblock - 1) / block34_per_threadblock;

//     int fill0_nnz_short13 = threadblock13 * block13_per_threadblock * MMA_M * MMA_K;
//     int fill0_nnz_short34 = threadblock34 * block34_per_threadblock * MMA_M * MMA_K;
//     int fill0_nnz_short22 = threadblock22 * block22_per_threadblock * MMA_M * MMA_K;
//     int fill0_nnz_short = ((short_row_1 + 1) / 2) * 2 + fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
//     half *short_val = (half *)malloc(sizeof(half) * fill0_nnz_short);
//     int *short_cid = (int *)malloc(sizeof(int) * fill0_nnz_short);
//     memset(short_val, 0.0, sizeof(half) * fill0_nnz_short);
//     memset(short_cid, 0, sizeof(int) * fill0_nnz_short);

//     #pragma omp parallel for
//     for (int i = 0; i < short_block13; i++) {
//         half* cur_short_val = short_val + i * MMA_M * MMA_K;
//         int *cur_short_cid = short_cid + i * MMA_M * MMA_K;
//         for (int j = 0; j < BlockSize && i * BlockSize + j < common_13; j++) {
//             int cur_row_1 = short_rid_1[short_row_1 + i * BlockSize + j];
//             int cur_row_3 = short_rid_3[i * BlockSize + j];
//             cur_short_val[j * MMA_K] = csrValA[csrRowPtrA[cur_row_1]];
//             cur_short_cid[j * MMA_K] = csrColIdxA[csrRowPtrA[cur_row_1]];
//             cur_short_val[j * MMA_K + 1] = csrValA[csrRowPtrA[cur_row_3]];
//             cur_short_val[j * MMA_K + 2] = csrValA[csrRowPtrA[cur_row_3] + 1];
//             cur_short_val[j * MMA_K + 3] = csrValA[csrRowPtrA[cur_row_3] + 2];
//             cur_short_cid[j * MMA_K + 1] = csrColIdxA[csrRowPtrA[cur_row_3]];
//             cur_short_cid[j * MMA_K + 2] = csrColIdxA[csrRowPtrA[cur_row_3] + 1];
//             cur_short_cid[j * MMA_K + 3] = csrColIdxA[csrRowPtrA[cur_row_3] + 2];
//         }
//     }

//     #pragma omp parallel for
//     for (int i = 0; i < short_row_3; i ++)
//     {
//         half *cur_short_val = short_val + fill0_nnz_short13 + i * MMA_K;
//         int *cur_short_cid = short_cid + fill0_nnz_short13 + i * MMA_K;
        
//         int cur_row = short_rid_3[common_13 + i];

//         cur_short_val[0] = csrValA[csrRowPtrA[cur_row]];
//         cur_short_val[1] = csrValA[csrRowPtrA[cur_row] + 1]; 
//         cur_short_val[2] = csrValA[csrRowPtrA[cur_row] + 2]; 
//         cur_short_cid[0] = csrColIdxA[csrRowPtrA[cur_row]];
//         cur_short_cid[1] = csrColIdxA[csrRowPtrA[cur_row] + 1]; 
//         cur_short_cid[2] = csrColIdxA[csrRowPtrA[cur_row] + 2]; 
//     }

//     #pragma omp parallel for
//     for (int i = 0; i < short_row_4; i ++)
//     {
//         MAT_VAL_TYPE *cur_short_val = short_val + fill0_nnz_short13 + (short_row_3 + i) * MMA_K;
//         int *cur_short_cid = short_cid + fill0_nnz_short13 + (short_row_3 + i) * MMA_K;
        
//         int cur_row = short_rid_4[i];

//         cur_short_val[0] = csrValA[csrRowPtrA[cur_row]];
//         cur_short_val[1] = csrValA[csrRowPtrA[cur_row] + 1]; 
//         cur_short_val[2] = csrValA[csrRowPtrA[cur_row] + 2]; 
//         cur_short_val[3] = csrValA[csrRowPtrA[cur_row] + 3]; 
//         cur_short_cid[0] = csrColIdxA[csrRowPtrA[cur_row]];
//         cur_short_cid[1] = csrColIdxA[csrRowPtrA[cur_row] + 1]; 
//         cur_short_cid[2] = csrColIdxA[csrRowPtrA[cur_row] + 2]; 
//         cur_short_cid[3] = csrColIdxA[csrRowPtrA[cur_row] + 3]; 
//     }

//     int group22 = (short_block22 + 3) / 4;
//     #pragma omp parallel for
//     for (int i = 0; i < group22; i ++)
//     {
//         MAT_VAL_TYPE *cur_short_val = short_val + fill0_nnz_short13 + fill0_nnz_short34 + i * 4 * MMA_M * MMA_K;
//         int *cur_short_cid = short_cid + fill0_nnz_short13 + fill0_nnz_short34 + i * 4 * MMA_M * MMA_K;

//         for (int j = 0; j < (BlockSize * 4 * 2) && (i * BlockSize * 4 * 2 + j) < short_row_2; j ++)
//         {
//             int cur_row = short_rid_2[i * BlockSize * 4 * 2 + j];
//             cur_short_val[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2] = csrValA[csrRowPtrA[cur_row]];
//             cur_short_val[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2 + 1] = csrValA[csrRowPtrA[cur_row] + 1];
//             cur_short_cid[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2] = csrColIdxA[csrRowPtrA[cur_row]];
//             cur_short_cid[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2 + 1] = csrColIdxA[csrRowPtrA[cur_row] + 1];
//         }
//     }

//     int offset_short_row1 = fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
//     #pragma omp parallel for
//     for (int i = 0; i < short_row_1; i ++)
//     {
//         int cur_row = short_rid_1[i];
//         short_val[offset_short_row1 + i] = csrValA[csrRowPtrA[cur_row]];
//         short_cid[offset_short_row1 + i] = csrColIdxA[csrRowPtrA[cur_row]];
//     }
//     radix_sort(rptA, ridA, row_block);
//     exclusive_scan(rptA, row_block + 1);
//     exclusive_scan(long_rpt, row_long + 1);
//     nnz_long = long_rpt[row_long];
//     memcpy(order_rid, long_rid, sizeof(int) * row_long);
//     memcpy(order_rid + row_long, ridA, sizeof(int) * row_block);
//     int group13 = common_13 / (4 * BlockSize);
//     #pragma omp parallel for
//     for (int i = 0; i < group13; i ++)
//     {
//         int *cur_order_rid = order_rid + row_long + row_block + i * BlockSize * 4 * 2;
//         for (int j = 0; j < BlockSize * 4; j ++)
//         {
//             cur_order_rid[j] = short_rid_1[short_row_1 + i * BlockSize * 4 + j];
//             cur_order_rid[BlockSize * 4 + j] = short_rid_3[i * BlockSize * 4 + j];
//         }
//     }
//     memcpy(order_rid + row_long + row_block + common_13 * 2, short_rid_3 + common_13, sizeof(int) * short_row_3);
//     memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3, short_rid_4, sizeof(int) * short_row_4);
//     memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3 + short_row_4, short_rid_2, sizeof(int) * short_row_2);
//     memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3 + short_row_4 + short_row_2, short_rid_1, sizeof(int) * short_row_1);
//     memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3 + short_row_4 + short_row_2 + short_row_1, zero_rid, sizeof(int) * row_zero);

// }