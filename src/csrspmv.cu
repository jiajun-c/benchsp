#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>
#include <cuda_fp16.h>
#include <chrono>
#include "matrixFormat.hpp"
#include <chrono>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
#include "utils.h"
using namespace std;
#define WARP 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128 
#define WARP_CAPACITY 16    

__global__ void spmv_fp64(int32_t *rowPtr, int32_t *colIdx, int32_t rows, int32_t nnz, double* values,double* x, double* outvalues) { 
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;
    // if (tid == 0) {
    //     printf("short rows: %d %d mapID:%d values:%f xval:%f\n", rowPtr[tid], rowPtr[tid+1], shortMap[tid], float( values[0]));
    // }
    double tmp = 0.0;
    int count = rowPtr[tid+1] - rowPtr[tid];
    if (count > 32) return;
    for (int32_t i = rowPtr[tid]; i < rowPtr[tid+1]; i++) {
        int32_t col = colIdx[i];
        double val = values[i];
        double xval = x[col];
        tmp += xval * val;
    }
    outvalues[tid] = tmp;
}

__global__ void get_rowPtrbyWarp_csr(int *d_blcPtr, int *rowPtrbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    rowPtrbyWarp[rowid] = (d_blcPtr[rowid + 1] - d_blcPtr[rowid] + WARP_CAPACITY - 1) / WARP_CAPACITY;
}


__global__ void get_rowIdxbyWarp_csr(int *rowPtrbyWarp, int *rowIdxbyWarp, int blc_row)
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

__global__ void spmv_fp64_balance(int32_t* rowPtr, int32_t *rowPtrbyWarp, int32_t *colIdx, int32_t* rowIdxbyWarp, int32_t rows, int32_t nnz, int32_t warp_num, double* values, double* x, double* d_y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= warp_num)
        return;
    int target_row_id = rowIdxbyWarp[tid];
    int start = rowPtr[target_row_id] + (tid - rowPtrbyWarp[target_row_id]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < rowPtr[target_row_id + 1] ? start + WARP_CAPACITY : rowPtr[target_row_id + 1];
    // if (end >= nnz)
    //     return;
    double res = 0;
    // printf("start:%d end:%d target_row_id:%d tid:%d\n",  rowPtrbyWarp[target_row_id],  rowPtrbyWarp[target_row_id+1], target_row_id, tid);
    // printf("start:%d end:%d\n", start, end );
    #pragma unroll
    for (int i = start; i < end; i++) {
        // printf("%d\n", i);
        res += values[i] * x[colIdx[i]];
    }
    atomicAdd(&d_y[target_row_id],res);
}

int spmv_fp64_warpper(int32_t *rowPtr, int32_t *colIdx, int32_t rows, int32_t cols, int32_t nnz, double* values,double* x, double* outvalues, int repeat) {    double *d_values, *d_x, *d_outvalues;
    cudaMalloc((void**)&d_values, sizeof(double)*nnz);
    cudaMalloc((void**)&d_x, sizeof(double)*cols);
    cudaMalloc((void**)&d_outvalues, sizeof(double)*rows);
    cudaMemcpy(d_values, values, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(double)*cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outvalues, outvalues, sizeof(double)*rows, cudaMemcpyHostToDevice);
    int32_t *d_rowPtr, *d_colIdx;
    cudaMalloc((void**)&d_rowPtr, sizeof(int32_t)*(rows+1));
    cudaMalloc((void**)&d_colIdx, sizeof(int32_t)*nnz);
    cudaMemcpy(d_rowPtr, rowPtr, sizeof(int32_t)*(rows+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, sizeof(int32_t)*nnz, cudaMemcpyHostToDevice);
    int threadsNum = 4*WARP;
    int blocksNum = (rows+threadsNum-1)/threadsNum;
    for (int i = 0; i < repeat; i++) {
        auto start = chrono::high_resolution_clock::now();
        spmv_fp64<<<blocksNum, threadsNum>>>(d_rowPtr, d_colIdx, rows, nnz, d_values, d_x, d_outvalues);
        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        auto eplapsed = end - start;
        std::cout << "csr spmv time: " << chrono::duration_cast<chrono::microseconds>(eplapsed).count() << "us" << std::endl;
    }
    cudaMemcpy(outvalues, d_outvalues, sizeof(double)*rows, cudaMemcpyDeviceToHost);
    return 0;
}

int spmv_fp64_balance_warpper(int32_t *rowPtr, int32_t *colIdx, int32_t rows, int32_t cols, int32_t nnz, double* values,double* x, double* outvalues, int repeat) {  
    double *d_values, *d_x, *d_outvalues;
    cudaMalloc((void**)&d_values, sizeof(double)*nnz);
    cudaMalloc((void**)&d_x, sizeof(double)*cols);
    cudaMalloc((void**)&d_outvalues, sizeof(double)*rows);
    cudaMemcpy(d_values, values, sizeof(double)*nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(double)*cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outvalues, outvalues, sizeof(double)*rows, cudaMemcpyHostToDevice);
    printf("rows:%d cols:%d nnz:%d\n", rows, cols, nnz);

    int32_t *d_rowPtr, *d_colIdx, *rowIdxbyWarp;
    cudaMalloc((void**)&d_rowPtr, sizeof(int32_t)*(rows+1));
    cudaMalloc((void**)&d_colIdx, sizeof(int32_t)*nnz);
    cudaMemcpy(d_rowPtr, rowPtr, sizeof(int32_t)*(rows+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, sizeof(int32_t)*nnz, cudaMemcpyHostToDevice);
    int *rowPtrbyWarp;
    cudaMalloc((void**)&rowPtrbyWarp, sizeof(int)*(rows+1));
    int ThreadNum = WARP * 4;
    int BlockNum = (rows + ThreadNum - 1) / ThreadNum;
    get_rowPtrbyWarp_csr<<<BlockNum, ThreadNum>>>(d_rowPtr, rowPtrbyWarp, rows);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, rowPtrbyWarp, rowPtrbyWarp + rows + 1, rowPtrbyWarp, 0);
    cudaDeviceSynchronize();
    int warpnum = 0;
    cudaMemcpy(&warpnum, (rowPtrbyWarp) + rows, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMalloc((void **)&rowIdxbyWarp, sizeof(int) * warpnum);

    get_rowIdxbyWarp_csr<<<BlockNum, ThreadNum>>>(rowPtrbyWarp, rowIdxbyWarp, rows);
    int threadsNum = 4*WARP;
    int blocksNum = (warpnum+threadsNum-1)/threadsNum;
    // printf("warpnum: %d threadsNum:%d blocksNum:%d\n", warpnum, threadsNum, blocksNum);
            CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
    for (int i = 0; i < repeat; i++) {
        auto start = chrono::high_resolution_clock::now();
        spmv_fp64_balance<<<blocksNum, threadsNum>>>(d_rowPtr, rowPtrbyWarp, d_colIdx, rowIdxbyWarp, rows, nnz, warpnum, d_values, d_x, d_outvalues);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        auto eplapsed = end - start;
        std::cout << "csr spmv time: " << chrono::duration_cast<chrono::microseconds>(eplapsed).count() << "us" << std::endl;
    }
    cudaMemcpy(outvalues, d_outvalues, sizeof(double)*rows, cudaMemcpyDeviceToHost);
    cudaFree(rowPtrbyWarp);
    cudaFree(rowIdxbyWarp);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_outvalues);
    return 0;
}

