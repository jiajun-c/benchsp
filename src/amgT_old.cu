// #include "amgT.h"
// #include <thrust/scan.h>
// #include <thrust/device_ptr.h>
// #include <thrust/execution_policy.h>
// #include "matrixFormat.hpp"

// #define MASK_SIZE 256
// #define WARP_SIZE 32
// #define BSR_N 4
// #define BSR_M 4
// #define WARP_CAPACITY 64
// #define WARP_NUM_SPMV 4

// #define setbit(x, y) x |= (1 << y)    // set the yth bit of x is 1
// #define clrbit(x, y) x &= ~(1 << y)   // set the yth bit of x is 0
// #define getbit(x, y) ((x) >> (y) & 1) // get the yth bit of x

// #define CHECK(call)                                                             \
// do                                                                              \
// {                                                                               \
//     const cudaError_t error_code = call;                                        \
//     if (error_code != cudaSuccess)                                              \
//     {                                                                           \
//         printf("CUDA Error:\n");                                                \
//         printf("     File:      %s\n", __FILE__);                               \
//         printf("     Line       %d:\n", __LINE__);                              \
//         printf("     Error code:%d\n", error_code);                             \
//         printf("     Error text:%s\n", cudaGetErrorString(error_code));         \
//         exit(1);                                                                \
//     }                                                                           \
// }while(0)      
// // 一个行上的任务数量，一个任务由一个warp进行执行
// template <typename T>
// __global__ void get_rowPtrByWarp(T *d_blcPtr, T *rowPtrbyWarp, int blc_row) {
//     T rowid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (rowid >= blc_row) return;
//     rowPtrbyWarp[rowid] = (d_blcPtr[rowid+1] - d_blcPtr[rowid] + WARP_CAPACITY - 1)/WARP_CAPACITY;
// }

// __device__ __forceinline__ void mma_m8n8k4(double *acc, double &frag_a, double &frag_b)
// {
//     asm volatile(
//         "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
//         " { %0, %1 }, "
//         " { %2 }, "
//         " { %3 }, "
//         " { %0, %1 };"
//         : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
// }

// // 每个任务所处于的行
// template <typename T>
// __global__ void get_rowIdxbyWarp(T *rowPtrbyWarp, T *rowIdxbyWarp, int blc_row)
// {
//     int rowid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (rowid >= blc_row)
//         return;

//     int offset = rowPtrbyWarp[rowid];
//     int stride = rowPtrbyWarp[rowid + 1] - rowPtrbyWarp[rowid];

//     for (int i = offset; i < (offset + stride); i++)
//     {
//         rowIdxbyWarp[i] = rowid;
//     }
// }

// template <typename T, typename Y>
// __global__ void bsr_spmv_balanced_tc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
//                                           T *d_blcPtr, T *d_blcCid, Y *d_blcVal,
//                                           Y *d_x, Y *d_y, int blc_row, int blc_col, int row, int col,
//                                           Y alpha) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int warpid = tid / WARP_SIZE;
//     int laneid = tid & (WARP_SIZE - 1);
//     int blc_rid = warpid;
//     if (warpid >= warp_num) return;
//     int target_block_row = rowIdxbyWarp[warpid];
//     int start = d_blcPtr[target_block_row] + (warpid - rowPtrbyWarp[target_block_row]) * WARP_CAPACITY;
//     int end = start + WARP_CAPACITY < d_blcPtr[target_block_row + 1] ? start + WARP_CAPACITY : d_blcPtr[target_block_row + 1];
//     Y fragA, fragB, fragC[2] = {0};
//     for (int i = start; i < end; i+=2) { 
//         Y *curval = d_blcVal + i * BSR_M * BSR_N;
//         fragA = (i + 1 >= end && laneid >= 16) ? 0 : curval[laneid];
//         int laneid_mod_4 = laneid & 3;
//         int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
//         fragB = d_x[xid + laneid_mod_4];

//         mma_m8n8k4(fragC, fragA, fragB);

//     }
//     fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
//     fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

//     if (laneid == 0)
//     {
//         int rowid = target_block_row * 4;
//         if (rowid < row)
//             atomicAdd(&d_y[rowid], fragC[0] * alpha);
//     }
//     if (laneid == 4)
//     {
//         int rowid = target_block_row * 4 + 1;
//         if (rowid < row)
//             atomicAdd(&d_y[rowid], fragC[1] * alpha);
//     }
//     if (laneid == 9)
//     {
//         int rowid = target_block_row * 4 + 2;
//         if (rowid < row)
//             atomicAdd(&d_y[rowid], fragC[0] * alpha);
//     }
//     if (laneid == 13)
//     {
//         int rowid = target_block_row * 4 + 3;
//         if (rowid < row)
//             atomicAdd(&d_y[rowid], fragC[1] * alpha);
//     }
// }
// template <typename MAT_PTR_TYPE, typename MAT_VAL_TYPE>
// __global__ void bsr_spmv_tc_fp64(MAT_PTR_TYPE *d_blcPtr, MAT_PTR_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
//                                  MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
//                                  int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int warpid = tid >> 5;
//     int laneid = tid & (WARP_SIZE - 1);

//     int blc_rid = warpid;
//     if (blc_rid >= blc_row)
//         return;

//     int start = d_blcPtr[blc_rid];
//     int end = d_blcPtr[blc_rid + 1];

//     MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
//     for (int i = start; i < end; i += 2)
//     {
//         MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_M * BSR_N;
//         fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

//         int laneid_mod_4 = laneid & 3;
//         int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
//         fragB = d_x[xid + laneid_mod_4];

//         mma_m8n8k4(fragC, fragA, fragB);
//     }

//     fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
//     fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

//     if (laneid == 0)
//     {
//         int rowid = blc_rid * 4;
//         if (rowid < row)
//             d_y[rowid] += fragC[0] * alpha;
//     }
//     if (laneid == 4)
//     {
//         int rowid = blc_rid * 4 + 1;
//         if (rowid < row)
//             d_y[rowid] += fragC[1] * alpha;
//     }
//     if (laneid == 9)
//     {
//         int rowid = blc_rid * 4 + 2;
//         if (rowid < row)
//             d_y[rowid] += fragC[0] * alpha;
//     }
//     if (laneid == 13)
//     {
//         int rowid = blc_rid * 4 + 3;
//         if (rowid < row)
//             d_y[rowid] += fragC[1] * alpha;
//     }
// }


// template <typename T, typename Y>
// __global__ void bsr_spmv_balanced_cc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
//                                           T *d_blcPtr, T *d_blcCid, unsigned short *d_blcMap, Y *d_blcVal,
//                                           Y *d_x, Y *d_y, int blc_row, int blc_col, int row, int col,
//                                           Y alpha) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int warpid = tid / WARP_SIZE;
//     int laneid = tid & (WARP_SIZE - 1);
//     int groupid = laneid >> 2;
//     int tid_in_group = laneid & 3;
//     if (warpid >= warp_num) return;
//     int target_block_row = rowIdxbyWarp[warpid];
//     int start = d_blcPtr[target_block_row] + (warpid - rowPtrbyWarp[target_block_row]) * WARP_CAPACITY;
//     int end = start + WARP_CAPACITY < d_blcPtr[target_block_row + 1] ? start + WARP_CAPACITY : d_blcPtr[target_block_row + 1];
//     Y res = 0;
//     // 8 is group size
//     for (int i = start + groupid; i < end; i += 8) {
//         Y* curPtr = d_blcVal + i * BSR_M * BSR_N;
//         unsigned short mapA = d_blcMap[i];
//         int offset_b = d_blcCid[i] * BSR_M;
//         for (int c = 0; c < BSR_N; c++) {
//             int idx = tid_in_group * BSR_N + c;
//             if (getbit(mapA, idx)) {
//                 res += curPtr[idx] * d_x[offset_b + c];
//             }
//         }
//     }
//     __syncwarp();
//     res += __shfl_down_sync(0xffffffff, res, 16);
//     res += __shfl_down_sync(0xffffffff, res, 8);
//     res += __shfl_down_sync(0xffffffff, res, 4);
//     if (laneid < 4) {
//         atomicAdd(&d_y[target_block_row * BSR_M + laneid], res * alpha);
//     }
// }

// template <typename T, typename Y>
// __global__ void bsr_spmv_cc_fp64(T *d_blcPtr, T *d_blcCid, unsigned short *d_blcMap, Y *d_blcVal,
//                                  Y *d_x, Y *d_y,
//                                  int blc_row, int blc_col, int row, int col, Y alpha) {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int warpid = tid >> 5;
//     int laneid = tid & (WARP_SIZE - 1);

//     int blc_rid = warpid;
//     if (blc_rid >= blc_row)
//         return;

//     int groupid = laneid >> 2;
//     int tid_in_group = laneid & 3;

//     int start = d_blcPtr[blc_rid];
//     int end = d_blcPtr[blc_rid + 1];

//     Y res = 0;
//     for (int i = start + groupid; i < end; i += 8)
//     {
//         Y *cur_val = d_blcVal + i * 16;
//         unsigned short mapA = d_blcMap[i];

//         int offset_b = d_blcCid[i] * BSR_N;

//         for (int c = 0; c < BSR_N; c++)
//         {
//             int idx = tid_in_group * BSR_N + c;

//             if (getbit(mapA, idx))
//             {
//                 res += cur_val[idx] * d_x[offset_b + c];
//             }
//         }
//     }
//     __syncwarp();

//     res += __shfl_down_sync(0xffffffff, res, 16);
//     res += __shfl_down_sync(0xffffffff, res, 8);
//     res += __shfl_down_sync(0xffffffff, res, 4);

//     if (laneid < 4)
//     {
//         d_y[blc_rid * BSR_M + laneid] += alpha * res;
//     }
// }

// __forceinline__ __device__ int sum_warp_shfl_int(int sum)
// {
// #pragma unroll
//     for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
//         sum += __shfl_xor_sync(0xffffffff, sum, mask);
//         // printf("sum %d\n", sum);
//     }
//     return sum;
// }
// template <typename T>
// __global__ void csr2bsr_get_ptr(T* d_csrptr, T* d_csridx, T* d_bsrptr, int brow, int bcol, int row, int col) {
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int laneid = tid & (WARP_SIZE - 1);
//     int warpid = tid / WARP_SIZE;
//     // if (idx >= 32) return;
//     __shared__ unsigned int mask[MASK_SIZE];

//     int rowid = bid * 4 + warpid;
//     // if (rowid >= row) return;

//     int start = d_csrptr[rowid >= row ? row : rowid] + laneid;
//     int end = d_csrptr[(rowid + 1) >= row ? row : (rowid + 1)];
//     // if (start >= end) return;
//     // if (end == 1) {
//         // printf("rowid %d, laneid %d, start %d, end %d\n", rowid, laneid, start, end);
//     // }
//     int sum = 0;

//     for (int i = 0; i < col; i += MASK_SIZE * 4 * 32)
//     {
//         int cur_end = (i + MASK_SIZE * 4 * 32) < col ? (i + MASK_SIZE * 4 * 32) : col;
//         for (int id = tid; id < MASK_SIZE; id += blockDim.x)
//         {
//             mask[id] = 0;
//         }
//         __syncthreads();

//         for (; start < end; start += WARP_SIZE)
//         {
//             int cid = d_csridx[start];
//             if (cid < cur_end)
//             {
//                 int key = (cid - i) / BSR_N;
//                 atomicOr(&(mask[key >> 5]), 1 << (key & 31));
//             }
//             else
//             {
//                 break;
//             }
//         }
//         __syncthreads();

//         for (int id = tid; id < MASK_SIZE; id += blockDim.x)
//         {
//             unsigned int cur_num = mask[id];
//             sum += __popc(cur_num);
//         }
//         __syncthreads();
//     }
//     // printf("tid %d, sum %d\n", tid, sum);
//     __shared__ int sums[WARP_SIZE];
//     __syncthreads();
//     sums[laneid] = sum;
//     __syncthreads();
//     // sum = sum_warp_shfl_int(sum);
//     // __syncthreads();
//     // printf("tid %d, sum %d\n", tid, sum);

//     if (laneid == 0)
//     {
//         int res = 0;
//         for (int i = 0; i < WARP_SIZE; i++)
//         {
//             res += sums[i];
//         }
//         printf("bid %d, sum %d tid:%d \n", bid, res, tid);

//         atomicAdd(&d_bsrptr[bid], res);
//     }
// }


// template <typename T>
// void CSR2BSR_step1(T* d_csrptr, T* d_csridx, T* d_bsrptr, int brow, int bcol, int row, int col) {
//     int threadNum = 4 * WARP_SIZE;
//     int blockNum = brow;
//     T* ptr = (T*)malloc(sizeof(T) * (brow+1));
//     csr2bsr_get_ptr<T><<<blockNum, threadNum>>>(d_csrptr, d_csridx, d_bsrptr, brow, bcol, row, col);
//     cudaDeviceSynchronize();
//     cudaMemcpy(ptr, d_bsrptr, sizeof(T) * (brow+1), cudaMemcpyDeviceToHost);
//     // for (int i = 0; i < brow+1; i++) {
//     //     printf("row %d sum %d\n", i, ptr[i]);
//     // }
//     thrust::exclusive_scan(thrust::device, d_bsrptr, d_bsrptr + (brow + 1), d_bsrptr, 0);
//     cudaDeviceSynchronize();
// }
// #define BIN_COUNT 7
// template <typename T>
// __global__ void csr2bsr_compute_bin(T *d_bsrptr, int brow, T *bin_offset) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= brow) {
//         return;
//     }
//     int len = d_bsrptr[idx + 1] - d_bsrptr[idx];
//     int bit = max(25 - __clz(len), 0);
//     atomicAdd(&bin_offset[bit], 1);
//     __syncthreads();
// }

// template <typename T>
// __global__ void csr2bsr_set_bin(T *d_bsrptr, T *bin_rowidx, T* bin_offset, T* bin_size, int  *max_num, int brow) {
// //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
// //     if (idx >= brow) return;
// //     int len = d_bsrptr[idx + 1] - d_bsrptr[idx];
// //     int bit = max(25 - __clz(len), 0);
// //     int count = 0;
// //     count = atomicAdd(&bin_size[bit], 1);
// //     bin_rowidx[bin_offset[bit] + count] = idx;
// //     atomicMax(max_num, len);

// // }
//     int rid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (rid >= brow)
//         return;

//     int cur_Cub = d_bsrptr[rid + 1] - d_bsrptr[rid];
//     int idx = 0;

//     if (cur_Cub < 128)
//     {
//         idx = atomicAdd(&bin_size[0], 1);
//         bin_rowidx[bin_offset[0] + idx] = rid;
//     }
//     else if (cur_Cub >= 128 && cur_Cub < 256)
//     {
//         idx = atomicAdd(&bin_size[1], 1);
//         bin_rowidx[bin_offset[1] + idx] = rid;
//     }
//     else if (cur_Cub >= 256 && cur_Cub < 512)
//     {
//         idx = atomicAdd(&bin_size[2], 1);
//         bin_rowidx[bin_offset[2] + idx] = rid;
//     }
//     else if (cur_Cub >= 512 && cur_Cub < 1024)
//     {
//         idx = atomicAdd(&bin_size[3], 1);
//         bin_rowidx[bin_offset[3] + idx] = rid;
//     }
//     else if (cur_Cub >= 1024 && cur_Cub < 2048)
//     {
//         idx = atomicAdd(&bin_size[4], 1);
//         bin_rowidx[bin_offset[4] + idx] = rid;
//     }
//     else if (cur_Cub >= 2048 && cur_Cub < 4096)
//     {
//         idx = atomicAdd(&bin_size[5], 1);
//         bin_rowidx[bin_offset[5] + idx] = rid;
//     }
//     else
//     {
//         idx = atomicAdd(&bin_size[6], 1);
//         bin_rowidx[bin_offset[6] + idx] = rid;
//         atomicMax(max_num, cur_Cub);
//     }
// }
// __device__ __host__ int BinarySearch2(int *arr, int left, int right, int target)
// {
//     int low = left;
//     int high = right;
//     int mid = 0;
//     while (low <= high)
//     {
//         mid = (low + high) / 2;
//         if (target < arr[mid])
//             high = mid - 1;
//         else if (target > arr[mid])
//             low = mid + 1;
//         else
//             return mid;
//     }
//     return -1;
// }
// // BLOCK_COUNT 表示行块的数目
// template <int BLOCK_COUNT, typename T, typename Y>
// __global__ void csr2bsr_getidx_large(int *bin_rowidx, int *bin_offset, int bin,
//                                T *d_csrptr, T *d_csridx, Y *d_csrval,
//                                T *d_bsrptr, T *d_bsridx, Y *d_bsrval, unsigned short *d_bsrmap,
//                                int brow, int bcol, int row, int col) {
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int laneid = tid % 32;
//     int warpid = tid / WARP_SIZE;
//     int bin_row_offset = bin_offset[bin] + bid;
//     if (bin_row_offset >= bin_offset[bin + 1])
//         return;

//     __shared__ int hashtable[4096];
//     __shared__ unsigned int maptable[4096];
//     __shared__ int nz_num[1];

//     int sum_len = 0;
//     int rowid = bin_rowidx[bin_row_offset];

//     int start1 = (rowid * 4 + warpid) < row ? (d_csrptr[rowid * 4 + warpid] + laneid) : d_csrptr[row];
//     int start2 = start1;
//     int end = (rowid * 4 + warpid + 1) < row ? d_csrptr[rowid * 4 + warpid + 1] : d_csrptr[row];

//     for (int i = 0; i < col; i += 4096 * 4)
//     {
//         int cur_end = (i + 4096 * 4) < col ? (i + 4096 * 4) : col;

//         if (tid == 0)
//         {
//             nz_num[0] = 0;
//         }

//         for (int id = tid; id < 4096; id += blockDim.x)
//         {
//             hashtable[id] = -1;
//         }

//         for (int id = tid; id < 4096; id += blockDim.x)
//         {
//             maptable[id] = 0;
//         }
//         __syncthreads();

//         for (; start1 < end; start1 += WARP_SIZE)
//         {
//             int cid = d_csridx[start1];
//             if (cid < cur_end)
//             {
//                 int key = cid / BSR_N;
//                 int hashadr = key & (4096 - 1);
//                 while (1)
//                 {
//                     int keyexist = hashtable[hashadr];
//                     if (keyexist == key)
//                     {
//                         atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
//                         break;
//                     }
//                     else if (keyexist == -1)
//                     {
//                         int idx = atomicCAS(hashtable + hashadr, -1, key);
//                         if (idx == -1)
//                         {
//                             atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
//                             break;
//                         }
//                     }
//                     else
//                     {
//                         hashadr = (hashadr + 1) & (4096 - 1);
//                     }
//                 }
//             }
//             else
//             {
//                 break;
//             }
//         }
//         __syncthreads();

//         if (tid < WARP_SIZE)
//         {
//             for (int id = tid; id < 4096; id += WARP_SIZE)
//             {
//                 unsigned int res_map = maptable[id];
//                 int res = hashtable[id];
//                 if (res != -1)
//                 {
//                     int ind = atomicAdd(&nz_num[0], 1);
//                     hashtable[ind] = res;
//                     maptable[ind] = res_map;
//                 }
//             }
//         }
//         __syncthreads();

//         int len = nz_num[0];

//         int offset = d_bsrptr[rowid] + sum_len;
//         int target, count;
//         unsigned int target_map;
//         unsigned short set_num = 0x0000ffff;
//         for (int id = tid; id < len; id += blockDim.x)
//         {
//             target = hashtable[id];
//             target_map = maptable[id];
//             count = 0;

//             for (int j = 0; j < len; j++)
//             {
//                 count += ((unsigned int)(hashtable[j] - target) >> 31);
//             }
//             d_bsridx[offset + count] = target;
//             d_bsrmap[offset + count] = target_map & set_num;
//         }
//         __syncthreads();

//         Y *cur_bsrval = d_bsrval + (offset * (BSR_M * BSR_N));
//         for (; start2 < end; start2 += WARP_SIZE)
//         {

//             T cid = d_csridx[start2];
//             if (cid < cur_end)
//             {
//                 Y val = d_csrval[start2];
//                 int bcid = cid / BSR_N;

//                 int offset_cid = BinarySearch2(d_bsridx + offset, 0, len, bcid);
//                 int offset_idx = (warpid * BSR_M) + (cid % BSR_N);
//                 cur_bsrval[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = val;
//             }
//             else
//             {
//                 break;
//             }
//         }

//         sum_len += len;
//         __syncthreads();
//     }

// }
// // BLOCK_COUNT 表示行块的数目
// template <int SM_SIZE, typename T, typename Y>
// __global__ void csr2bsr_getidx(int *bin_rowidx, int *bin_offset, int bin,
//                                T *d_csrptr, T *d_csridx, Y *d_csrval,
//                                T *d_bsrptr, T *d_bsridx, Y *d_bsrval, unsigned short *d_bsrmap,
//                                int brow, int bcol, int row, int col) {
//   int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     int laneid = tid & (WARP_SIZE - 1);
//     int warpid = tid / WARP_SIZE;
//     int bin_row_offset = bin_offset[bin] + bid;
//     if (bin_row_offset >= bin_offset[bin + 1])
//         return;

//     __shared__ int hashtable[SM_SIZE];
//     __shared__ unsigned int maptable[SM_SIZE];
//     __shared__ int nz_num[1];

//     if (tid == 0)
//     {
//         nz_num[0] = 0;
//     }

//     for (int i = tid; i < SM_SIZE; i += blockDim.x)
//     {
//         hashtable[i] = -1;
//     }

//     for (int i = tid; i < SM_SIZE; i += blockDim.x)
//     {
//         maptable[i] = 0;
//     }
//     __syncthreads();

//     int rowid = bin_rowidx[bin_row_offset];

//     int start = (rowid * 4 + warpid) < row ? d_csrptr[rowid * 4 + warpid] : d_csrptr[row];
//     int end = (rowid * 4 + warpid + 1) < row ? d_csrptr[rowid * 4 + warpid + 1] : d_csrptr[row];

//     for (int j = start + laneid; j < end; j += WARP_SIZE)
//     {
//         int cid = d_csridx[j];
//         int key = cid / BSR_N;
//         int hashadr = key & (SM_SIZE - 1);
//         while (1)
//         {
//             int keyexist = hashtable[hashadr];
//             if (keyexist == key)
//             {
//                 atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
//                 break;
//             }
//             else if (keyexist == -1)
//             {
//                 int idx = atomicCAS(hashtable + hashadr, -1, key);
//                 if (idx == -1)
//                 {
//                     atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
//                     break;
//                 }
//             }
//             else
//             {
//                 hashadr = (hashadr + 1) & (SM_SIZE - 1);
//             }
//         }
//     }
//     __syncthreads();

//     if (tid < WARP_SIZE)
//     {
//         for (int i = tid; i < SM_SIZE; i += WARP_SIZE)
//         {
//             unsigned int res_map = maptable[i];
//             int res = hashtable[i];
//             if (res != -1)
//             {
//                 int ind = atomicAdd(&nz_num[0], 1);
//                 hashtable[ind] = res;
//                 maptable[ind] = res_map;
//             }
//         }
//     }
//     __syncthreads();

//     int len = nz_num[0];

//     int offset = d_bsrptr[rowid];
//     int target, count;
//     unsigned int target_map;
//     unsigned short set_num = 0x0000ffff;
//     for (int i = tid; i < len; i += blockDim.x)
//     {
//         target = hashtable[i];
//         target_map = maptable[i];
//         count = 0;

//         for (int j = 0; j < len; j++)
//         {
//             count += ((unsigned int)(hashtable[j] - target) >> 31);
//         }
//         d_bsridx[offset + count] = target;
//         d_bsrmap[offset + count] = target_map & set_num;
//     }
//     __syncthreads();

//     Y *cur_bsrval = d_bsrval + (offset * (BSR_M * BSR_N));
//     for (int j = start + laneid; j < end; j += WARP_SIZE)
//     {
//         T cid = d_csridx[j];
//         Y val = d_csrval[j];
//         int bcid = cid / BSR_N;

//         int offset_cid = BinarySearch2(d_bsridx + offset, 0, len, bcid);
//         int offset_idx = (warpid * BSR_M) + (cid % BSR_N);
//         cur_bsrval[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = val;
//     }
//     __syncthreads();
// }


// template <typename T, typename Y>
// void CSR2BSR_step2(T *d_csrptr, T *d_csridx, Y *d_csrval,
//                    T *d_bsrptr, T *d_bsridx, Y *d_bsrval, unsigned short *d_bsrmap,
//                    int brow, int bcol, int nnb, int row, int col) {
//     int* bin_offset, *bin_size;
//     cudaMalloc((void**)&bin_offset, (BIN_COUNT + 1) * sizeof(int));
//     cudaMalloc((void**)&bin_size, BIN_COUNT * sizeof(int));
//     cudaMemset(bin_offset, 0, (BIN_COUNT + 1) * sizeof(int));
//     cudaMemset(bin_size, 0, BIN_COUNT * sizeof(int));

//     T* bin_rowidx;
//     cudaMalloc((void**)&bin_rowidx, brow * sizeof(T));
//     int *max_num;
//     cudaMalloc((void**)&max_num, sizeof(int));

//     int threadNum = 4 * WARP_SIZE;
//     int blockNum = (brow + threadNum - 1) / threadNum;
//     // 获取到不同数量的行块的数目
//     csr2bsr_compute_bin<T><<<blockNum, threadNum>>>(d_bsrptr, brow, bin_offset);
//     cudaDeviceSynchronize();
//     thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + BIN_COUNT + 1, bin_offset, 0);

//     csr2bsr_set_bin<T><<<blockNum, threadNum>>>(d_csrptr, bin_rowidx, bin_offset, bin_size, max_num, brow);
//     cudaDeviceSynchronize();

//     int max_len;
//     cudaMemcpy(&max_len, max_num, sizeof(int), cudaMemcpyDeviceToHost);
//     int *offset = (int *)malloc(sizeof(int) * (BIN_COUNT + 1));
//     cudaMemcpy(offset, bin_offset, sizeof(int) * (BIN_COUNT + 1), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();
//     for (int i = BIN_COUNT - 1; i >= 0; i--) {
//         int row_num = offset[i + 1] - offset[i];
//         threadNum = WARP_SIZE * 4;
//         blockNum = row_num;
//         if (row_num) {
//             switch (i)
//             {
//             case 0:
//                 csr2bsr_getidx<128, T, Y><<<blockNum, threadNum>>>(bin_rowidx, bin_offset, i,
//                                                              d_csrptr, d_csridx, d_csrval,
//                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
//                                                              brow, bcol, row, col);
//                 break;
//             case 1:
//                 csr2bsr_getidx<256, T, Y><<<blockNum, threadNum>>>(bin_rowidx, bin_offset, i,
//                                                              d_csrptr, d_csridx, d_csrval,
//                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
//                                                              brow, bcol, row, col);
//                 break;
//             case 2:
//                 csr2bsr_getidx<512, T, Y><<<blockNum, threadNum>>>(bin_rowidx, bin_offset, i,
//                                                              d_csrptr, d_csridx, d_csrval,
//                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
//                                                              brow, bcol, row, col);
//                 break;
//             case 3:
//                 csr2bsr_getidx<1024, T, Y><<<blockNum, threadNum>>>(bin_rowidx, bin_offset, i,
//                                                              d_csrptr, d_csridx, d_csrval,
//                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
//                                                              brow, bcol, row, col);
//                 break;
//             case 4:
//                 csr2bsr_getidx<2048, T, Y><<<blockNum, threadNum>>>(bin_rowidx, bin_offset, i,
//                                                              d_csrptr, d_csridx, d_csrval,
//                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
//                                                              brow, bcol, row, col);
//                 break;
//             case 5:
//                 csr2bsr_getidx<4096, T, Y><<<blockNum, threadNum>>>(bin_rowidx, bin_offset, i,
//                                                              d_csrptr, d_csridx, d_csrval,
//                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
//                                                              brow, bcol, row, col);
//                 break;
//             case 6:
//                 csr2bsr_getidx_large<4096, T, Y><<<blockNum, threadNum>>>(bin_rowidx, bin_offset, i,
//                                                              d_csrptr, d_csridx, d_csrval,
//                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
//                                                              brow, bcol, row, col);
//                 break;
//             default:
//                 break;
//             }
//         }
//     }
// }

// template <typename T, typename Y>
// __global__ void getStand(T* rowPtr, double *sum, double avg_len, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     __shared__ Y partialSum[256];
//     if (idx < N) {
//         partialSum[threadIdx.x] = rowPtr[idx];
//     } else {
//         partialSum[threadIdx.x] = 0;
//     }
//     __syncthreads();
//     int i = blockDim.x / 2;
//     while (i != 0)
//     {
//         if (threadIdx.x < i)
//         {
//             partialSum[threadIdx.x] += partialSum[threadIdx.x + i];
//         }
//         __syncthreads();
//         i /= 2;
//     }

//     if (threadIdx.x == 0)
//     {
//         atomicAdd(&sum[0], partialSum[0]);
//     }
// }

// template <typename T, typename Y>
// void CSR2BSR(T *h_csrptr, T *h_csridx, Y *h_csrval, int nnz, int row, int col, bsrMAT<T, Y> *bsr) {
//     bsr->row = row;
//     bsr->col = col;
//     bsr->blc_row = (col + BSR_M - 1)/BSR_M;
//     bsr->blc_col = (row + BSR_N - 1)/BSR_N;
//     T* d_csrptr, *d_csridx;
//     Y* d_csrval;
//     // for (int i = 0; i <= row; i++) {
//     //     printf("%d ", h_csrptr[i]);
//     // }
//     // printf("\n");

//     cudaMalloc((void**)&d_csrptr, sizeof(T) * (row + 1));
//     cudaMalloc((void**)&d_csridx, sizeof(T) * nnz);
//     cudaMalloc((void**)&d_csrval, sizeof(Y) * nnz);

//     cudaMemcpy(d_csrptr, h_csrptr, sizeof(T) * (row + 1), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_csridx, h_csridx, sizeof(T) * nnz, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_csrval, h_csrval, sizeof(Y) * nnz, cudaMemcpyHostToDevice);
    
//     cudaMalloc((void**)&(bsr->blcPtr), sizeof(T) * (bsr->blc_row + 1));
//     cudaMemset(bsr->blcPtr, 0, sizeof(T) * (bsr->blc_row + 1));
//     CSR2BSR_step1(d_csrptr, d_csridx, bsr->blcPtr, bsr->blc_row, bsr->blc_col, row, col);
//     cudaMemcpy(&(bsr->blc_num), &bsr->blcPtr[bsr->blc_row], sizeof(T), cudaMemcpyDeviceToHost);

//     // bsr 的非零元数目
//     bsr->nnz = bsr->blc_num * BSR_M * BSR_N;
    
//     // 平均每个块中非零元数目
//     bsr->avg_nnz = double(nnz) / double(bsr->blc_num);

//     double *result_gpu;
//     cudaMalloc((void**)&result_gpu, sizeof(double));
//     cudaMemset(result_gpu, 0.0, sizeof(double));
//     int thread_num_stand = 256;
//     int block_num_stand = (bsr->blc_row + thread_num_stand - 1) / thread_num_stand;
//     double avg_len = (double)bsr->blc_num / (double)bsr->blc_row;

//     getStand<T, Y><<<block_num_stand, thread_num_stand>>>(bsr->blcPtr, result_gpu, avg_len, bsr->blc_row);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&bsr->stand, result_gpu, sizeof(double), cudaMemcpyDeviceToHost);
//     bsr->stand = sqrtf(bsr->stand / bsr->blc_row);
//     bsr->avg_nnz = nnz / double(bsr->blc_num);
//     printf("blc_num: %d, avg_nnz: %f, stand: %f\n", bsr->blc_num, bsr->avg_nnz, bsr->stand);

//     cudaMalloc((void **)&bsr->blcIdx, sizeof(T) * bsr->blc_num);
//     cudaMalloc((void **)&bsr->blcVal, sizeof(Y) * bsr->nnz);
//     cudaMalloc((void **)&bsr->blcMap, sizeof(unsigned short) * (bsr->blc_num + 1));

//     cudaMemset(bsr->blcVal, 0, sizeof(Y) * bsr->nnz);
//     cudaMemset(bsr->blcMap, 0, sizeof(unsigned short) * (bsr->blc_num + 1));

//     CSR2BSR_step2<T, Y>(d_csrptr, d_csridx, d_csrval,
//                       bsr->blcPtr, bsr->blcIdx, bsr->blcVal, bsr->blcMap,
//                       bsr->blc_row, bsr->blc_col, bsr->blc_num, bsr->row, bsr->col);
//     // make work balance
//     cudaMalloc((void **)&bsr->rowPtrbyWarp, sizeof(T) * (bsr->blc_row + 1));
//     cudaMemset(bsr->rowPtrbyWarp, 0, sizeof(T) * (bsr->blc_row + 1));
//     int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
//     int BlockNum = (bsr->blc_row + ThreadNum - 1) / ThreadNum;

//     get_rowPtrByWarp<T><<<BlockNum, ThreadNum>>>(bsr->blcPtr, bsr->rowPtrbyWarp, bsr->blc_row);
//     cudaDeviceSynchronize();

//     thrust::exclusive_scan(thrust::device, bsr->rowPtrbyWarp, bsr->rowPtrbyWarp + (bsr->blc_row + 1), bsr->rowPtrbyWarp, 0);
//     cudaDeviceSynchronize();
//     cudaMemcpy(&bsr->warpnum, (bsr->rowPtrbyWarp) + bsr->blc_row, sizeof(T), cudaMemcpyDeviceToHost);

//     cudaMalloc((void **)&bsr->rowIdxbyWarp, sizeof(T) * bsr->warpnum);

//     get_rowIdxbyWarp<T><<<BlockNum, ThreadNum>>>(bsr->rowPtrbyWarp, bsr->rowIdxbyWarp, bsr->blc_row);
//     cudaDeviceSynchronize();
// }

// int amgT_spmv_fp64(int32_t *hA_csrOffsets, int32_t *hA_columns, double *hA_values, double *hX, double* hY, int32_t A_num_rows, int32_t A_num_cols, int32_t A_nnz, int repeat) {
//     bsrMAT<int32_t, double> bsr;
//     // printf("CSR2BSR start\n");
//     CSR2BSR(hA_csrOffsets, hA_columns, hA_values, A_nnz, A_num_rows, A_num_cols, &bsr);
//     double stand = bsr.stand;
//     double avgnz = bsr.avg_nnz;
//     int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
//     int BlockNum_b = (bsr.warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
//     int BlockNum = (bsr.blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
//     int BlockNum2 = (bsr.row + ThreadNum - 1) / ThreadNum;
//     double alpha = 1.0;
//     double *dvecX, *dvecY;
//     cudaMalloc((void **)&dvecX, sizeof(double) * A_num_cols);
//     cudaMalloc((void **)&dvecY, sizeof(double) * A_num_rows);
//     cudaMemcpy(dvecX, hX, sizeof(double) * bsr.row, cudaMemcpyHostToDevice);
//     cudaMemcpy(dvecY, hY, sizeof(double) * bsr.row, cudaMemcpyHostToDevice);
//     // printf("stand : %f avgnz : %f\n", stand, avgnz);
//     for (int i = 0; i < repeat; i++) {
//         auto start = std::chrono::high_resolution_clock::now(); 
//         // if (stand >= 12 && avgnz >= 10)
//         // {
//         //     // ===tensor core, balanced===
//         //     bsr_spmv_balanced_tc_fp64<int32_t, double><<<BlockNum_b, ThreadNum>>>(bsr.rowPtrbyWarp, bsr.rowIdxbyWarp, bsr.warpnum, bsr.blcPtr, bsr.blcIdx, bsr.blcVal, dvecX, dvecY, bsr.blc_row, bsr.blc_col, bsr.row, bsr.col, alpha);
//         //     cudaDeviceSynchronize();
//         //     // ===============================
//         // }
//         // else if (stand >= 12 && avgnz < 10)
//         // {
//         //     // ===cuda core, balanced===
//         //     bsr_spmv_balanced_cc_fp64<int32_t, double><<<BlockNum_b, ThreadNum>>>(bsr.rowPtrbyWarp, bsr.rowIdxbyWarp, bsr.warpnum, bsr.blcPtr, bsr.blcIdx, bsr.blcMap, bsr.blcVal, dvecX, dvecY, bsr.blc_row, bsr.blc_col, bsr.row, bsr.col, alpha);
//         //     cudaDeviceSynchronize();
//         //     // ===============================
//         // }
//         // else if (stand < 12 && avgnz >= 10)
//         // {
//             // ===tensor core===
//             bsr_spmv_tc_fp64<int32_t, double><<<BlockNum, ThreadNum>>>(bsr.blcPtr, bsr.blcIdx, bsr.blcVal, dvecX, dvecY, bsr.blc_row, bsr.blc_col, bsr.row, bsr.col, alpha);
//             cudaDeviceSynchronize();
//             // ===============================
//         // }
//         // else
//         // {
//         //     // ===cuda core===
//         //     // printf("use cuda core :%d %d\n", BlockNum, ThreadNum);
//         //     bsr_spmv_cc_fp64<int32_t, double><<<BlockNum, ThreadNum>>>(bsr.blcPtr, bsr.blcIdx, bsr.blcMap, bsr.blcVal, dvecX, dvecY, bsr.blc_row, bsr.blc_col, bsr.row, bsr.col, alpha);
//         //     cudaDeviceSynchronize();
//         //     // ===============================
//         // }
//         auto end = std::chrono::high_resolution_clock::now(); 
//         auto elapsed = end - start;
//         std::cout << "baseline 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
//     }
//     cudaMemcpy(hY, dvecY, sizeof(double) * bsr.row, cudaMemcpyDeviceToHost);
//     return 0;
// }
