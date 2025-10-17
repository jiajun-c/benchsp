#include "amgT.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "matrixFormat.hpp"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
#include "utils.h"
#define ADAPTIVE_AMGT_SPMV
#define MASK_SIZE 256
#define WARP_SIZE 32
#define BSR_N 4
#define BSR_M 4
#define BSR_NNZ 16
#define WARP_CAPACITY 64
#define WARP_NUM_SPMV 4
#define HYPRE_Real double
#define setbit(x, y) x |= (1 << y)    // set the yth bit of x is 1
#define clrbit(x, y) x &= ~(1 << y)   // set the yth bit of x is 0
#define getbit(x, y) ((x) >> (y) & 1) // get the yth bit of x

#define MAT_VAL_TYPE double
#define MAT_PTR_TYPE int
#define MAT_IDX_TYPE int
#define MAT_MAP_TYPE unsigned short
__forceinline__ __device__ int sum_warp_shfl_int(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
        // printf("sum %d\n", sum);
    }
    return sum;
}
__device__ __host__ int BinarySearch2(int *arr, int left, int right, int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__device__ __forceinline__ void mma_m8n8k4(MAT_VAL_TYPE *acc, MAT_VAL_TYPE &frag_a, MAT_VAL_TYPE &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}

__device__ __host__ int BinarySearch2_SpMV(int *arr, int left, int right, int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__global__ void bsr_spmv(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                         MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                         int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha, MAT_VAL_TYPE beta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[0] + beta * d_y[rowid];
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[1] + beta * d_y[rowid];
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[0] + beta * d_y[rowid];
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[1] + beta * d_y[rowid];
    }
}



__global__ void bsr_spmv_balanced_cc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, MAT_VAL_TYPE *d_blcVal,
                                          MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);
    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;
    // warp_num = 357588/64;
    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE res = 0;
    // int read_count = 0, write_count = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];
        int offset_b = d_blcCid[i] * BSR_N;
        // read_count++;
        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        atomicAdd(&d_y[blc_rid * BSR_M + laneid], res * alpha);
    }
    // if (tid < 100) {
    // printf("tid:%d %d\n", tid, read_count);
    // }
}
__global__ void bsr_spmv_tc_fp64(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                                 MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                                 int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha)
{   
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];
    // __shared__ MAT_VAL_TYPE sharedA[1024];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    // for (int i = start; i < end; i += 2)
    // {
    //         // MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;

    //     fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];
    //     sharedA[i * 32 + laneid] = fragA;
    // }
    #pragma unroll
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];
        // fragA = sharedA[i * 32 + laneid];
        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
}

__global__ void bsr_spmv_cc_fp64(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, MAT_VAL_TYPE *d_blcVal,
                                 MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                                 int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE res = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        d_y[blc_rid * BSR_M + laneid] += alpha * res;
    }
}

__global__ void bsr_spmv_balanced_tc_fp64_dbsr(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                                          MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;

    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];
        MAT_IDX_TYPE *cur_idx = d_blcCid + i * BSR_NNZ ;
        fragB = (i + 1 >= end && laneid >= 16) ? 0 : d_x[cur_idx[laneid]];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
    // if (blc_rid*4 >= 220) {
    //     printf("err\n");
    // }
    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
}


__global__ void bsr_spmv_balanced_tc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                                          MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;

    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
}

__global__ void get_rowPtrbyWarp(MAT_PTR_TYPE *d_blcPtr, int *rowPtrbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    rowPtrbyWarp[rowid] = (d_blcPtr[rowid + 1] - d_blcPtr[rowid] + WARP_CAPACITY - 1) / WARP_CAPACITY;
}

__global__ void get_rowIdxbyWarp(int *rowPtrbyWarp, int *rowIdxbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    int offset = rowPtrbyWarp[rowid];
    int stride = rowPtrbyWarp[rowid + 1] - rowPtrbyWarp[rowid];

    for (int i = offset; i < (offset + stride); i++)
    {
        rowIdxbyWarp[i] = rowid;
    }
}
__global__ void getStand(MAT_PTR_TYPE *rowptr, double *sum, double avg_len, int N)
{

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ MAT_VAL_TYPE partialSum[256];

    if (idx < N)
    {
        // partialSum[threadIdx.x] = a[idx] * b[idx];
        partialSum[threadIdx.x] = pow(rowptr[idx + 1] - rowptr[idx] - avg_len, 2);
    }
    else
    {
        partialSum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&sum[0], partialSum[0]);
    }
}



#define MASK_SIZE 256

__global__ void csr2bsr_get_ptr(MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_PTR_TYPE *d_bsrptr,
                                int brow, int bcol, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int laneid = tid & (WARP_SIZE - 1);
    int warpid = tid / WARP_SIZE;

    __shared__ unsigned int mask[MASK_SIZE];

    int rowid = bid * 4 + warpid;
    // if (rowid >= row) return;

    int start = d_csrptr[rowid >= row ? row : rowid] + laneid;
    int end = d_csrptr[(rowid + 1) >= row ? row : (rowid + 1)];

    int sum = 0;

    for (int i = 0; i < col; i += MASK_SIZE * 4 * 32)
    {
        int cur_end = (i + MASK_SIZE * 4 * 32) < col ? (i + MASK_SIZE * 4 * 32) : col;
        for (int id = tid; id < MASK_SIZE; id += blockDim.x)
        {
            mask[id] = 0;
        }
        __syncthreads();

        for (; start < end; start += WARP_SIZE)
        {
            int cid = d_csridx[start];
            if (cid < cur_end)
            {
                int key = (cid - i) / BSR_N;
                atomicOr(&(mask[key >> 5]), 1 << (key & 31));
            }
            else
            {
                break;
            }
        }
        __syncthreads();

        for (int id = tid; id < MASK_SIZE; id += blockDim.x)
        {
            unsigned int cur_num = mask[id];
            sum += __popc(cur_num);
        }
        __syncthreads();
    }
    // __shared__ MAT_VAL_TYPE sums[WARP_SIZE];
    sum = sum_warp_shfl_int(sum);
    __syncthreads();

    if (laneid == 0)
    {   
        atomicAdd(&d_bsrptr[bid], sum);
    }
}

#define CONVERT_BIN 7

__global__ void csr2bsr_compute_bin(MAT_PTR_TYPE *d_bsrptr, int brow, int *bin_offset)
{
    int rid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rid >= brow)
        return;

    int len = d_bsrptr[rid + 1] - d_bsrptr[rid];

    if (len < 128)
    {
        atomicAdd(&bin_offset[0], 1);
    }
    else if (len >= 128 && len < 256)
    {
        atomicAdd(&bin_offset[1], 1);
    }
    else if (len >= 256 && len < 512)
    {
        atomicAdd(&bin_offset[2], 1);
    }
    else if (len >= 512 && len < 1024)
    {
        atomicAdd(&bin_offset[3], 1);
    }
    else if (len >= 1024 && len < 2048)
    {
        atomicAdd(&bin_offset[4], 1);
    }
    else if (len >= 2048 && len < 4096)
    {
        atomicAdd(&bin_offset[5], 1);
    }
    else
    {
        atomicAdd(&bin_offset[6], 1);
    }
    __syncthreads();
}

__global__ void csr2bsr_set_bin(MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *bin_rowidx, int *bin_offset, int *bin_size, int *max_num, int brow)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= brow)
        return;

    int cur_Cub = d_bsrptr[rid + 1] - d_bsrptr[rid];
    int idx = 0;

    if (cur_Cub < 128)
    {
        idx = atomicAdd(&bin_size[0], 1);
        bin_rowidx[bin_offset[0] + idx] = rid;
    }
    else if (cur_Cub >= 128 && cur_Cub < 256)
    {
        idx = atomicAdd(&bin_size[1], 1);
        bin_rowidx[bin_offset[1] + idx] = rid;
    }
    else if (cur_Cub >= 256 && cur_Cub < 512)
    {
        idx = atomicAdd(&bin_size[2], 1);
        bin_rowidx[bin_offset[2] + idx] = rid;
    }
    else if (cur_Cub >= 512 && cur_Cub < 1024)
    {
        idx = atomicAdd(&bin_size[3], 1);
        bin_rowidx[bin_offset[3] + idx] = rid;
    }
    else if (cur_Cub >= 1024 && cur_Cub < 2048)
    {
        idx = atomicAdd(&bin_size[4], 1);
        bin_rowidx[bin_offset[4] + idx] = rid;
    }
    else if (cur_Cub >= 2048 && cur_Cub < 4096)
    {
        idx = atomicAdd(&bin_size[5], 1);
        bin_rowidx[bin_offset[5] + idx] = rid;
    }
    else
    {
        idx = atomicAdd(&bin_size[6], 1);
        bin_rowidx[bin_offset[6] + idx] = rid;
        atomicMax(max_num, cur_Cub);
    }
}

template <int SM_SIZE>
__global__ void csr2bsr_getidx(int *bin_rowidx, int *bin_offset, int bin,
                               MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_VAL_TYPE *d_csrval,
                               MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *d_bsridx, MAT_VAL_TYPE *d_bsrval, MAT_MAP_TYPE *d_bsrmap,
                               int brow, int bcol, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int laneid = tid & (WARP_SIZE - 1);
    int warpid = tid / WARP_SIZE;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    __shared__ int hashtable[SM_SIZE];
    __shared__ unsigned int maptable[SM_SIZE];
    __shared__ int nz_num[1];

    if (tid == 0)
    {
        nz_num[0] = 0;
    }

    for (int i = tid; i < SM_SIZE; i += blockDim.x)
    {
        hashtable[i] = -1;
    }

    for (int i = tid; i < SM_SIZE; i += blockDim.x)
    {
        maptable[i] = 0;
    }
    __syncthreads();

    int rowid = bin_rowidx[bin_row_offset];

    int start = (rowid * 4 + warpid) < row ? d_csrptr[rowid * 4 + warpid] : d_csrptr[row];
    int end = (rowid * 4 + warpid + 1) < row ? d_csrptr[rowid * 4 + warpid + 1] : d_csrptr[row];

    for (int j = start + laneid; j < end; j += WARP_SIZE)
    {
        int cid = d_csridx[j];
        int key = cid / BSR_N;
        int hashadr = key & (SM_SIZE - 1);
        while (1)
        {
            int keyexist = hashtable[hashadr];
            if (keyexist == key)
            {
                atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                break;
            }
            else if (keyexist == -1)
            {
                int idx = atomicCAS(hashtable + hashadr, -1, key);
                if (idx == -1)
                {
                    atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                    break;
                }
            }
            else
            {
                hashadr = (hashadr + 1) & (SM_SIZE - 1);
            }
        }
    }
    __syncthreads();

    if (tid < WARP_SIZE)
    {
        for (int i = tid; i < SM_SIZE; i += WARP_SIZE)
        {
            unsigned int res_map = maptable[i];
            int res = hashtable[i];
            if (res != -1)
            {
                int ind = atomicAdd(&nz_num[0], 1);
                hashtable[ind] = res;
                maptable[ind] = res_map;
            }
        }
    }
    __syncthreads();

    int len = nz_num[0];

    int offset = d_bsrptr[rowid];
    int target, count;
    unsigned int target_map;
    unsigned short set_num = 0x0000ffff;
    for (int i = tid; i < len; i += blockDim.x)
    {
        target = hashtable[i];
        target_map = maptable[i];
        count = 0;

        for (int j = 0; j < len; j++)
        {
            count += ((unsigned int)(hashtable[j] - target) >> 31);
        }
        d_bsridx[offset + count] = target;
        d_bsrmap[offset + count] = target_map & set_num;
    }
    __syncthreads();

    MAT_VAL_TYPE *cur_bsrval = d_bsrval + (offset * (BSR_M * BSR_N));
    for (int j = start + laneid; j < end; j += WARP_SIZE)
    {
        MAT_IDX_TYPE cid = d_csridx[j];
        MAT_VAL_TYPE val = d_csrval[j];
        int bcid = cid / BSR_N;

        int offset_cid = BinarySearch2(d_bsridx + offset, 0, len, bcid);
        int offset_idx = (warpid * BSR_M) + (cid % BSR_N);
        cur_bsrval[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = val;
    }
    __syncthreads();
}

__global__ void csr2bsr_getidx_large(int *bin_rowidx, int *bin_offset, int bin,
                                     MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_VAL_TYPE *d_csrval,
                                     MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *d_bsridx, MAT_VAL_TYPE *d_bsrval, MAT_MAP_TYPE *d_bsrmap,
                                     int brow, int bcol, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int laneid = tid & (WARP_SIZE - 1);
    int warpid = tid / WARP_SIZE;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    __shared__ int hashtable[4096];
    __shared__ unsigned int maptable[4096];
    __shared__ int nz_num[1];

    int sum_len = 0;

    int rowid = bin_rowidx[bin_row_offset];

    int start1 = (rowid * 4 + warpid) < row ? (d_csrptr[rowid * 4 + warpid] + laneid) : d_csrptr[row];
    int start2 = start1;
    int end = (rowid * 4 + warpid + 1) < row ? d_csrptr[rowid * 4 + warpid + 1] : d_csrptr[row];

    for (int i = 0; i < col; i += 4096 * 4)
    {
        int cur_end = (i + 4096 * 4) < col ? (i + 4096 * 4) : col;

        if (tid == 0)
        {
            nz_num[0] = 0;
        }

        for (int id = tid; id < 4096; id += blockDim.x)
        {
            hashtable[id] = -1;
        }

        for (int id = tid; id < 4096; id += blockDim.x)
        {
            maptable[id] = 0;
        }
        __syncthreads();

        for (; start1 < end; start1 += WARP_SIZE)
        {
            int cid = d_csridx[start1];
            if (cid < cur_end)
            {
                int key = cid / BSR_N;
                int hashadr = key & (4096 - 1);
                while (1)
                {
                    int keyexist = hashtable[hashadr];
                    if (keyexist == key)
                    {
                        atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                        break;
                    }
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
                            atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                            break;
                        }
                    }
                    else
                    {
                        hashadr = (hashadr + 1) & (4096 - 1);
                    }
                }
            }
            else
            {
                break;
            }
        }
        __syncthreads();

        if (tid < WARP_SIZE)
        {
            for (int id = tid; id < 4096; id += WARP_SIZE)
            {
                unsigned int res_map = maptable[id];
                int res = hashtable[id];
                if (res != -1)
                {
                    int ind = atomicAdd(&nz_num[0], 1);
                    hashtable[ind] = res;
                    maptable[ind] = res_map;
                }
            }
        }
        __syncthreads();

        int len = nz_num[0];

        int offset = d_bsrptr[rowid] + sum_len;
        int target, count;
        unsigned int target_map;
        unsigned short set_num = 0x0000ffff;
        for (int id = tid; id < len; id += blockDim.x)
        {
            target = hashtable[id];
            target_map = maptable[id];
            count = 0;

            for (int j = 0; j < len; j++)
            {
                count += ((unsigned int)(hashtable[j] - target) >> 31);
            }
            d_bsridx[offset + count] = target;
            d_bsrmap[offset + count] = target_map & set_num;
        }
        __syncthreads();

        MAT_VAL_TYPE *cur_bsrval = d_bsrval + (offset * (BSR_M * BSR_N));
        for (; start2 < end; start2 += WARP_SIZE)
        {

            MAT_IDX_TYPE cid = d_csridx[start2];
            if (cid < cur_end)
            {
                MAT_VAL_TYPE val = d_csrval[start2];
                int bcid = cid / BSR_N;

                int offset_cid = BinarySearch2(d_bsridx + offset, 0, len, bcid);
                int offset_idx = (warpid * BSR_M) + (cid % BSR_N);
                cur_bsrval[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = val;
            }
            else
            {
                break;
            }
        }

        sum_len += len;
        __syncthreads();
    }
}

void CSR2BSR_step1(MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_PTR_TYPE *d_bsrptr,
                   int brow, int bcol, int row, int col)
{
    int ThreadNum = 4 * WARP_SIZE;
    int BlockNum = brow;
    csr2bsr_get_ptr<<<BlockNum, ThreadNum>>>(d_csrptr, d_csridx, d_bsrptr, brow, bcol, row, col);
    cudaDeviceSynchronize();
    
}

void CSR2BSR_step2(MAT_PTR_TYPE *d_bsrptr, int brow)
{
    thrust::exclusive_scan(thrust::device, d_bsrptr, d_bsrptr + (brow + 1), d_bsrptr, 0);
    MAT_PTR_TYPE *h_bsrptr;
    h_bsrptr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (brow + 1));
    cudaMemcpy(h_bsrptr, d_bsrptr, sizeof(MAT_PTR_TYPE) * (brow + 1), cudaMemcpyDeviceToHost);
    // for (int i = 1; i < brow; i++) {
    //     int count = h_bsrptr[i] - h_bsrptr[i - 1];
    //     if (count > 16) {
    //         printf("count: %d\n", count);
    //     }
    //     // printf("%d\n", h_bsrptr[i] - h_bsrptr[i - 1]);
    // }
    cudaDeviceSynchronize();
}

void CSR2BSR_step3(MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_VAL_TYPE *d_csrval,
                   MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *d_bsridx, MAT_VAL_TYPE *d_bsrval, MAT_MAP_TYPE *d_bsrmap,
                   int brow, int bcol, int nnb, int row, int col)
{
    int *bin_offset, *bin_size;
    cudaMalloc((void **)&bin_offset, sizeof(int) * (CONVERT_BIN + 1));
    cudaMalloc((void **)&bin_size, sizeof(int) * CONVERT_BIN);
    cudaMemset(bin_offset, 0, sizeof(int) * (CONVERT_BIN + 1));
    cudaMemset(bin_size, 0, sizeof(int) * CONVERT_BIN);

    MAT_IDX_TYPE *bin_rowidx;
    cudaMalloc((void **)&bin_rowidx, sizeof(MAT_IDX_TYPE) * brow);
    int *max_num;
    cudaMalloc((void **)&max_num, sizeof(int));

    int ThreadNum = WARP_SIZE * 4;
    int BlockNum = (brow + ThreadNum - 1) / ThreadNum;

    csr2bsr_compute_bin<<<BlockNum, ThreadNum>>>(d_bsrptr, brow, bin_offset);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + (CONVERT_BIN + 1), bin_offset, 0);

    csr2bsr_set_bin<<<BlockNum, ThreadNum>>>(d_bsrptr, bin_rowidx, bin_offset, bin_size, max_num, brow);
    cudaDeviceSynchronize();

    int max_len;
    cudaMemcpy(&max_len, max_num, sizeof(int), cudaMemcpyDeviceToHost);
    int *offset = (int *)malloc(sizeof(int) * (CONVERT_BIN + 1));
    cudaMemcpy(offset, bin_offset, sizeof(int) * (CONVERT_BIN + 1), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = CONVERT_BIN - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * 4;
        BlockNum = row_num;

        if (row_num)
        {
            switch (i)
            {
            case 0:
                csr2bsr_getidx<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                             d_csrptr, d_csridx, d_csrval,
                                                             d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                             brow, bcol, row, col);
                break;
            case 1:
                csr2bsr_getidx<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                             d_csrptr, d_csridx, d_csrval,
                                                             d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                             brow, bcol, row, col);
                break;
            case 2:
                csr2bsr_getidx<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                             d_csrptr, d_csridx, d_csrval,
                                                             d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                             brow, bcol, row, col);
                break;
            case 3:
                csr2bsr_getidx<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            case 4:
                csr2bsr_getidx<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            case 5:
                csr2bsr_getidx<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            case 6:
            {
                csr2bsr_getidx_large<<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            }
            }
            cudaDeviceSynchronize();
        }
    }
}

void CSR2BSR_GPU(bsrMAT *bsrmat, int32_t *hA_csrOffsets, int32_t *hA_columns, double *hA_values, double *hX, double* hY, int32_t A_num_rows, int32_t A_num_cols, int32_t A_nnz) {
{
        int *d_csrptr;
        int *d_csridx;
        cudaMalloc((void**)&d_csrptr, sizeof(int) * (A_num_rows + 1));
        cudaMalloc((void**)&d_csridx, sizeof(int) * A_nnz);

        MAT_VAL_TYPE *d_csrval;
        cudaMalloc((void**)&d_csrval, sizeof(MAT_VAL_TYPE) * A_nnz);

        CHECK(cudaMemcpy(d_csrptr, hA_csrOffsets, sizeof(int) * (A_num_rows + 1), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_csridx, hA_columns, sizeof(int) * A_nnz, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_csrval, hA_values, sizeof(MAT_VAL_TYPE) * A_nnz, cudaMemcpyHostToDevice));
        bsrmat->row = A_num_rows;
        bsrmat->col = A_num_cols;
        bsrmat->blc_row = (bsrmat->row + BSR_M - 1) / BSR_M;
        bsrmat->blc_col = (bsrmat->col + BSR_N - 1) / BSR_N;

        // csr2bsr step 1: get the block number of each block-row
        cudaMalloc((void **)&(bsrmat->blcPtr), sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        cudaMemset(bsrmat->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        CSR2BSR_step1(d_csrptr, d_csridx, bsrmat->blcPtr, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col);
        // csr2bsr step1 over
        // csr2bsr step 2: pre-sum, get the bsrPtr array
        CSR2BSR_step2(bsrmat->blcPtr, bsrmat->blc_row);

        // csr2bsr step2 over

        cudaMemcpy(&(bsrmat->blc_num), &bsrmat->blcPtr[bsrmat->blc_row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
        bsrmat->nnz = bsrmat->blc_num * BSR_M * BSR_N;
        bsrmat->avg_nnz = (double)A_nnz/ (double)(bsrmat->blc_num);
        printf("avg_nnz: %f\n",bsrmat->avg_nnz );
        HYPRE_Real *result_gpu;
        cudaMalloc((void **)&result_gpu, sizeof(HYPRE_Real));
        cudaMemset(result_gpu, 0.0, sizeof(HYPRE_Real));
        int thread_num_stand = 256;
        int block_num_stand = (bsrmat->blc_row + thread_num_stand - 1) / thread_num_stand;
        double avg_len = (double)bsrmat->blc_num / (double)bsrmat->blc_row;
        printf("avg_len = %f\n", avg_len);
        getStand<<<block_num_stand, thread_num_stand>>>(bsrmat->blcPtr, result_gpu, avg_len, bsrmat->blc_row);
        cudaDeviceSynchronize();
        cudaMemcpy(&bsrmat->stand, result_gpu, sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

        bsrmat->stand = sqrtf(bsrmat->stand / bsrmat->blc_row);
        
        // csr2bsr step 3: get the blcIdx, blcVal, blcMap
        cudaMalloc((void **)&bsrmat->blcIdx, sizeof(MAT_IDX_TYPE) * bsrmat->blc_num);
        cudaMalloc((void **)&bsrmat->blcVal, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
        cudaMalloc((void **)&bsrmat->blcMap, sizeof(MAT_MAP_TYPE) * (bsrmat->blc_num + 1));

        cudaMemset(bsrmat->blcVal, 0, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
        cudaMemset(bsrmat->blcMap, 0, sizeof(MAT_MAP_TYPE) * (bsrmat->blc_num + 1));

        CSR2BSR_step3(d_csrptr, d_csridx, d_csrval,
                      bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, bsrmat->blcMap,
                      bsrmat->blc_row, bsrmat->blc_col, bsrmat->blc_num, bsrmat->row, bsrmat->col);
        // csr2bsr step3 over
    }
}

void BSR_BALANCED_PREPROCESS_GPU(bsrMAT *bsrmat)
{
    // bsrmat->stand = 12;
#ifdef ADAPTIVE_AMGT_SPMV
    if (bsrmat->stand >= 12)
#endif
    {
        // load balanced preprocess
        printf("load balanced preprocess\n");
        cudaMalloc((void **)&bsrmat->rowPtrbyWarp, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        cudaMemset(bsrmat->rowPtrbyWarp, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));

        int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
        int BlockNum = (bsrmat->blc_row + ThreadNum - 1) / ThreadNum;

        get_rowPtrbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->rowPtrbyWarp, bsrmat->blc_row);
        cudaDeviceSynchronize();
        thrust::exclusive_scan(thrust::device, bsrmat->rowPtrbyWarp, bsrmat->rowPtrbyWarp + bsrmat->blc_row + 1, bsrmat->rowPtrbyWarp, 0);
        cudaDeviceSynchronize();

        cudaMemcpy(&bsrmat->warpnum, (bsrmat->rowPtrbyWarp) + bsrmat->blc_row, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);

        cudaMalloc((void **)&bsrmat->rowIdxbyWarp, sizeof(int) * bsrmat->warpnum);

        get_rowIdxbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->blc_row);
        cudaDeviceSynchronize();
    }
}

int amgT_spmv_fp64_dbsr(dbsrMat *dbsr, double *hX, double *hY, int32_t repeat) {
    dbsrMat d_dbsr;
    double *dX, *dY;
    cudaMalloc((void **)&dX, sizeof(double) * dbsr->row);
    cudaMalloc((void **)&dY, sizeof(double) * dbsr->row);
    cudaMalloc((void **)&d_dbsr.blcPtr, sizeof(int) * (dbsr->blc_row + 1));
    cudaMalloc((void **)&d_dbsr.colIdx, sizeof(int) * dbsr->blocknnz);
    cudaMalloc((void **)&d_dbsr.blcVal, sizeof(double) * dbsr->blocknnz);
    cudaMemcpy(d_dbsr.blcPtr, dbsr->blcPtr, sizeof(int) * (dbsr->blc_row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dbsr.colIdx, dbsr->colIdx, sizeof(int) * dbsr->blocknnz, cudaMemcpyHostToDevice);
    for (int i = 0; i < dbsr->blocknnz; i++) {
        if (dbsr->colIdx[i] >= 5300) {
            printf("colIdx[%d] = %d\n", i, dbsr->colIdx[i]);
        }
    }

    cudaMemcpy(d_dbsr.blcVal, dbsr->blcVal, sizeof(double) * dbsr->blocknnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, sizeof(double) * dbsr->row, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, sizeof(double) * dbsr->row, cudaMemcpyHostToDevice);

    d_dbsr.blc_row = dbsr->blc_row;
    printf("load balanced preprocess\n");
    cudaMalloc((void **)&d_dbsr.rowPtrbyWarp, sizeof(MAT_PTR_TYPE) * (d_dbsr.blc_row + 1));
    cudaMemset(d_dbsr.rowPtrbyWarp, 0, sizeof(MAT_PTR_TYPE) * (d_dbsr.blc_row + 1));
    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum = (d_dbsr.blc_row + ThreadNum - 1) / ThreadNum;

    get_rowPtrbyWarp<<<BlockNum, ThreadNum>>>(d_dbsr.blcPtr, d_dbsr.rowPtrbyWarp, d_dbsr.blc_row);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, d_dbsr.rowPtrbyWarp, d_dbsr.rowPtrbyWarp + d_dbsr.blc_row + 1, d_dbsr.rowPtrbyWarp, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(&d_dbsr.warpnum, (d_dbsr.rowPtrbyWarp) + d_dbsr.blc_row, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
    printf("d_dbsr warpnum = %d\n", d_dbsr.warpnum);
    cudaMalloc((void **)&d_dbsr.rowIdxbyWarp, sizeof(int) * d_dbsr.warpnum);
    double alpha = 1.0;
    int BlockNum_b = (d_dbsr.warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;

    get_rowIdxbyWarp<<<BlockNum, ThreadNum>>>(d_dbsr.rowPtrbyWarp, d_dbsr.rowIdxbyWarp, d_dbsr.blc_row);
    for (int i = 0; i < repeat; i++) {
        auto start = std::chrono::steady_clock::now();
        bsr_spmv_balanced_tc_fp64_dbsr<<<BlockNum_b, ThreadNum>>>(d_dbsr.rowPtrbyWarp, d_dbsr.rowIdxbyWarp, d_dbsr.warpnum, d_dbsr.blcPtr, d_dbsr.colIdx, d_dbsr.blcVal, dX, dY,d_dbsr.blc_row, d_dbsr.blc_col, d_dbsr.row,d_dbsr.col, alpha);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        auto elapsed = end - start;
        std::cout << "dbsr amgT 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    }
    cudaMemcpy(hY, dY, sizeof(double) * dbsr->row, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    return 0;
}

int amgT_spmv_fp64(int32_t *hA_csrOffsets, int32_t *hA_columns, double *hA_values, double *hX, double* hY, int32_t A_num_rows, int32_t A_num_cols, int32_t A_nnz, int repeat)
{
    bsrMAT bsrmat;
    printf("=== block matrix info ====\n");

    CSR2BSR_GPU(&bsrmat, hA_csrOffsets, hA_columns, hA_values, hX, hY, A_num_rows, A_num_cols, A_nnz);
    BSR_BALANCED_PREPROCESS_GPU(&bsrmat);
    // return 0;

    printf("bsrmat->warpnum = %d blc_num %d\n", bsrmat.warpnum, bsrmat.blc_num);
    MAT_VAL_TYPE *dvecX;
    MAT_VAL_TYPE *dvecY;
    cudaMalloc((void **)&dvecX, sizeof(MAT_VAL_TYPE) *bsrmat.col);
    cudaMalloc((void **)&dvecY, sizeof(MAT_VAL_TYPE) * bsrmat.row);
    cudaMemcpy(dvecX, hX, sizeof(MAT_VAL_TYPE) * bsrmat.col, cudaMemcpyHostToDevice);
    cudaMemcpy(dvecY, hY, sizeof(MAT_VAL_TYPE) * bsrmat.row, cudaMemcpyHostToDevice);
    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_b = (bsrmat.warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum = (bsrmat.blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    // int BlockNum2 = (bsrmat.row + ThreadNum - 1) / ThreadNum;
    // printf("ThreadNum = %d BlockNum = %d BlockNum_b:%d BlockNum2 = %d\n", ThreadNum, BlockNum, BlockNum_b, BlockNum2);
    // if (beta != 1)
    // {
    //     beta_vecY<<<BlockNum2, ThreadNum>>>(dvecY, beta, bsrmat->row);
    //     cudaDeviceSynchronize();
    // }
    cudaDeviceSynchronize();
    double stand =bsrmat.stand;
    double avgnz = bsrmat.avg_nnz;
    double alpha = 1.0;
    // for (int i = 0; i < repeat; i++) {
    printf("stand = %f avgnz = %f\n", stand, avgnz);
    printf("blc_row = %d\n", bsrmat.blc_row);
    printf("blc_col = %d\n", bsrmat.blc_col);
    printf("blc_num = %d\n", bsrmat.blc_num);
    printf("=== block matrix end ====\n");

    for (int i = 0; i < repeat; i++) {
    auto start = std::chrono::steady_clock::now();
    // stand = 12;
#ifdef ADAPTIVE_AMGT_SPMV
    if (stand >= 12 && avgnz >= 10)
    {
        // ===tensor core, balanced===
        bsr_spmv_balanced_tc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat.rowPtrbyWarp, bsrmat.rowIdxbyWarp, bsrmat.warpnum, bsrmat.blcPtr,bsrmat.blcIdx, bsrmat.blcVal, dvecX, dvecY,bsrmat.blc_row, bsrmat.blc_col, bsrmat.row,bsrmat.col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand >= 12 && avgnz < 10)
    {
        // ===cuda core, balanced===
        bsr_spmv_balanced_cc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat.rowPtrbyWarp, bsrmat.rowIdxbyWarp, bsrmat.warpnum, bsrmat.blcPtr, bsrmat.blcIdx, bsrmat.blcMap, bsrmat.blcVal, dvecX, dvecY, bsrmat.blc_row, bsrmat.blc_col, bsrmat.row, bsrmat.col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand < 12 && avgnz >= 10)
    {
        // ===tensor core===
        // printf("===tensor core===\n");
        bsr_spmv_tc_fp64<<<BlockNum, ThreadNum>>>(bsrmat.blcPtr, bsrmat.blcIdx, bsrmat.blcVal, dvecX, dvecY, bsrmat.blc_row,bsrmat.blc_col,bsrmat.row, bsrmat.col, alpha);
        // bsr_spmv_cc_fp64<<<BlockNum, ThreadNum>>>(bsrmat.blcPtr, bsrmat.blcIdx, bsrmat.blcMap, bsrmat.blcVal, dvecX, dvecY, bsrmat.blc_row,bsrmat.blc_col, bsrmat.row, bsrmat.col, alpha);

        cudaDeviceSynchronize();
        // ===============================
    }
    else
    {
        // ===cuda core===
        bsr_spmv_cc_fp64<<<BlockNum, ThreadNum>>>(bsrmat.blcPtr, bsrmat.blcIdx, bsrmat.blcMap, bsrmat.blcVal, dvecX, dvecY, bsrmat.blc_row,bsrmat.blc_col, bsrmat.row, bsrmat.col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }

#else
    printf("===tensor core, balanced===\n");
    bsr_spmv_balanced_tc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat.rowPtrbyWarp, bsrmat.rowIdxbyWarp, bsrmat.warpnum, bsrmat.blcPtr, bsrmat.blcIdx, bsrmat.blcVal, dvecX, dvecY, bsrmat.blc_row, bsrmat.blc_col, bsrmat.row, bsrmat.col, alpha);
#endif
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = end - start;
    std::cout << "amgT 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    }
    cudaMemcpy(hY, dvecY, sizeof(MAT_VAL_TYPE) * bsrmat.row, cudaMemcpyDeviceToHost);
    cudaFree(dvecX);
    cudaFree(dvecY);
    cudaDeviceSynchronize();
    return 0;
}