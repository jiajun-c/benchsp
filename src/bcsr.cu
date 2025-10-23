#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include "../utils/utils.hpp"
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define CHECK(call)                                                             \
do                                                                              \
{                                                                               \
    const cudaError_t error_code = call;                                        \
    if (error_code != cudaSuccess)                                              \
    {                                                                           \
        printf("CUDA Error:\n");                                                \
        printf("     File:      %s\n", __FILE__);                               \
        printf("     Line       %d:\n", __LINE__);                              \
        printf("     Error code:%d\n", error_code);                             \
        printf("     Error text:%s\n", cudaGetErrorString(error_code));         \
        exit(1);                                                                \
    }                                                                           \
}while(0)               

__global__ void bcsr_spmv_kernel(
    int *row_ptr,
    int *col_idx,
    double *Aval,
    int n,
    double *x,
    double *y
) {
    __shared__ double s_out[256];
    int tid = threadIdx.x;
    int target_block_row = (threadIdx.x + blockDim.x * blockIdx.x)/32;
    int lane = tid % 32;
    int first_block;
    int last_block;
    int target_block;
    int c, r, col, stride;
    double local_out;
    double x_elem;
    double A_elem;
    s_out[tid] = 0.0;
    int bs = 2;
    first_block = row_ptr[target_block_row];
    last_block = row_ptr[target_block_row + 1];
    target_block = first_block + lane/(bs*bs);
    c = (lane / bs) % bs;
    r = lane % bs;
    if (target_block_row < n) {
        if (lane < (32/(bs*bs))*(bs*bs)) {
            local_out = 0.0;
            for (; target_block < last_block; target_block += 32/(bs*bs)) {
                col = col_idx[target_block];
                x_elem = x[col * bs + c];
                A_elem = Aval[target_block*bs*bs + c*bs + r];
                local_out += x_elem * A_elem;
            }
            s_out[tid] = local_out;
            __syncthreads();
            stride = ((32 / bs) / 2); // only for bs=3 case
            for (; stride >= 1; stride /= 2) {
                if (lane < stride*bs && lane + stride*bs < 32) {
                    __syncthreads();
                    s_out[tid] += s_out[tid + stride * bs];
                }
            }
            if (lane < bs)
            {
                y[target_block_row * bs + lane] = s_out[tid];
            }
        }
    }
}
struct shared_memory
{
  __device__ inline operator double *()
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};
__device__ unsigned int round_up_to_power_of_two(unsigned int n) {
    if (n == 0) return 1;  // 处理n=0的情况，返回2^0=1
    n--;                    // 如果n已是2的幂，减1后高位右移
    n |= n >> 1;            // 将最高位1扩散到所有低位
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;           // 覆盖32位整数的所有位
    return n + 1;           // 加1得到2的幂
}

__global__ void bcsr_spmv_kernel_column_by_column_part1(
    int64_t n_block_row,
    int64_t *row_ptr,
    int64_t *col_idx,
    double *Aval,
    int64_t n,
    double *x,
    double *y
) {
    // printf("called\n");
    // return;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int64_t block_row = idx / 32;
    int64_t bs = 3;
    // printf("block_row = %d, lane = %d n_block_row=%d\n", block_row, lane, n_block_row);
    __shared__ double partial_sums[512];
    if (block_row < n_block_row/2) {
        const int64_t first_block = row_ptr[block_row];
        const int64_t last_block = row_ptr[block_row + 1];
        int64_t col = first_block *bs + lane/bs;
        int r = lane % bs;
        // double *partial_sums = shared_memory(); ///< Size is equal to blockDim.x * sizeof(double)
        double local_out = 0.0;
        if (lane < (32/bs)*bs) {
            // printf("col: %d last_block:%d\n", col, last_block * bs );
            for (; col < last_block * bs; col += 32/bs) {
                const int block = col / bs;
                const int c = col % bs;
                const double value = Aval[block * bs * bs + r*bs + c];
                const double x_value = x[col_idx[block] * bs + c];
                local_out += x_value * value;
                // printf("col = %d, r = %d, value = %f, x_value = %f, local_out = %f x_idx: %d \n", col, r, value, x_value, local_out, col_idx[block] * bs + r);
            }

            partial_sums[threadIdx.x] = local_out;
            __syncthreads();
                            // printf("stride = %d\n", stride);

            for (int stride = round_up_to_power_of_two((32 / bs) / 2); stride >= 1; stride /= 2)
            {
                __syncthreads();
                if ((lane < stride * bs) && ((lane + stride * bs) < 32))
                partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
            }
                        // __syncthreads();

            // if (threadIdx.x == 0) {
            //     for (int i = 0; i < 32; i++) {
            //         printf("partial_sums[%d] = %f\n", i, partial_sums[i]);
            //     }
            // }
            if (lane < bs)
            {
                y[block_row * bs + lane] = partial_sums[threadIdx.x];
                // printf("y[%d,%d] = %f id:%d tid:%d\n", block_row, lane, partial_sums[threadIdx.x], block_row * bs + lane, threadIdx.x);

            }
            // return;
        }
    }
}

__global__ void bcsr_spmv_kernel_column_by_column(
    int64_t n_block_row,
    int64_t *row_ptr,
    int64_t *col_idx,
    double *Aval,
    int64_t n,
    double *x,
    double *y
) {
    // printf("called\n");
    // return;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int64_t block_row = idx / 32;
    int64_t bs = 3;
    // printf("block_row = %d, lane = %d n_block_row=%d\n", block_row, lane, n_block_row);
    __shared__ double partial_sums[512];
    if (block_row < n_block_row) {
        const int64_t first_block = row_ptr[block_row];
        const int64_t last_block = row_ptr[block_row + 1];
        int64_t col = first_block *bs + lane/bs;
        int r = lane % bs;
        // double *partial_sums = shared_memory(); ///< Size is equal to blockDim.x * sizeof(double)
        double local_out = 0.0;
        if (lane < (32/bs)*bs) {
            // printf("col: %d last_block:%d\n", col, last_block * bs );
            for (; col < last_block * bs; col += 32/bs) {
                const int block = col / bs;
                const int c = col % bs;
                const double value = Aval[block * bs * bs + r*bs + c];
                const double x_value = x[col_idx[block] * bs + c];
                local_out += x_value * value;
                // printf("col = %d, r = %d, value = %f, x_value = %f, local_out = %f x_idx: %d \n", col, r, value, x_value, local_out, col_idx[block] * bs + r);
            }

            partial_sums[threadIdx.x] = local_out;
            __syncthreads();
                            // printf("stride = %d\n", stride);

            for (int stride = round_up_to_power_of_two((32 / bs) / 2); stride >= 1; stride /= 2)
            {
                __syncthreads();
                if ((lane < stride * bs) && ((lane + stride * bs) < 32))
                partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
            }
                        // __syncthreads();

            // if (threadIdx.x == 0) {
            //     for (int i = 0; i < 32; i++) {
            //         printf("partial_sums[%d] = %f\n", i, partial_sums[i]);
            //     }
            // }
            if (lane < bs)
            {
                y[block_row * bs + lane] = partial_sums[threadIdx.x];
                // printf("y[%d,%d] = %f id:%d tid:%d\n", block_row, lane, partial_sums[threadIdx.x], block_row * bs + lane, threadIdx.x);

            }
            // return;
        }
    }
}

void bcsr_spmv_fp64(int64_t *hA_csrOffsets, int64_t *hA_columns, double *hA_values32, double *hX32, double* hY32, int64_t A_num_rows, int64_t A_num_cols, int64_t A_nnz, int repeat) {
    CSRFormat<double, int64_t>csr;
    csr.rowPtr = hA_csrOffsets;
    csr.colIdx = hA_columns;
    csr.row = A_num_rows;
    csr.col = A_num_cols;
    csr.values = (double*)malloc(A_nnz * sizeof(double));
    for (int i = 0; i < A_nnz; i++) {
        csr.values[i] = double(hA_values32[i]);
    }
    cudaSetDevice(0); 

    BCSRFormat<double, int64_t>bcsr;
    bcsr.TILE_M = 3;
    bcsr.TILE_N = 3;
    bcsr.TILE_K = 3;
    bcsr.nnz = A_nnz;

    csrToBcsr<double, int64_t>(&csr, &bcsr);
    // printf("called\n");

    // free(bcsr.values);
    // free(bcsr.bcsrcolIdx);
    // free(bcsr.bcsrRowPtr);
    // free(bcsr.relativeBlockIndexMapping);
    // free(csr.values);
    // free(csr.colIdx);
    // free(csr.rowPtr);
    // printf("end\n");
    // return;
    // for (int i = 0; i < 18; i++) {
    //     printf("bcsr.values[%d] = %f\n", i, bcsr.values[i]);
    // }
    int64_t *dbcsrRowPtr;
    cudaMalloc(&dbcsrRowPtr, (sizeof(int64_t)*(bcsr.rowRegions + 1)));
    cudaMemcpy(dbcsrRowPtr, bcsr.bcsrRowPtr, (sizeof(int64_t)*(bcsr.rowRegions + 1)), cudaMemcpyHostToDevice);

    int64_t *dbcsrColIdx;
    cudaMalloc(&dbcsrColIdx, (sizeof(int64_t)*bcsr.nonzeroBlocks));
    cudaMemcpy(dbcsrColIdx, bcsr.bcsrcolIdx, (sizeof(int64_t)*bcsr.nonzeroBlocks), cudaMemcpyHostToDevice);

    double *dbcsrValues;
    cudaMalloc(&dbcsrValues, (sizeof(double)*bcsr.nonzeroBlocks*bcsr.TILE_M*bcsr.TILE_K));
    cudaMemcpy(dbcsrValues, bcsr.values, (sizeof(double)*bcsr.nonzeroBlocks*bcsr.TILE_M*bcsr.TILE_K), cudaMemcpyHostToDevice);

    double *dx;
    cudaMalloc(&dx, (sizeof(double)*csr.col));
    // for (int i = 0; i < 16; i++) {
    //     printf("hX32[%d] = %f\n", i, hX32[i]);

    // }
    cudaMemcpy(dx, hX32, (sizeof(double)*csr.col), cudaMemcpyHostToDevice);

    double *dy;
    cudaMalloc(&dy, (sizeof(double)*csr.row));
    cudaMemcpy(dy, hY32, (sizeof(double)*csr.row), cudaMemcpyHostToDevice);
    CHECK(cudaGetLastError());
    for (int i = 0; i < repeat; i++) {
        auto start = std::chrono::high_resolution_clock::now(); 
        // cudaStream_t stream1, stream2;
        // cudaStreamCreate(&stream1);
        // cudaStreamCreate(&stream2);
        // bcsr_spmv_kernel_column_by_column_part1<<<bcsr.rowRegions/2,THREADS_PER_BLOCK, 0, stream1>>>(bcsr.rowRegions, dbcsrRowPtr, dbcsrColIdx, dbcsrValues, bcsr.row, dx, dy);
        bcsr_spmv_kernel_column_by_column<<<bcsr.rowRegions,THREADS_PER_BLOCK>>>(bcsr.rowRegions, dbcsrRowPtr, dbcsrColIdx, dbcsrValues, bcsr.row, dx, dy);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now(); 
        auto elapsed = end - start;
        std::cout << "bcsr 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    }
    CHECK(cudaGetLastError());
    cudaMemcpy(hY32, dy, (sizeof(double)*csr.row), cudaMemcpyDeviceToHost);

    cudaFree(dbcsrRowPtr);
    cudaFree(dbcsrColIdx);
    cudaFree(dbcsrValues);
    cudaFree(dx);
    cudaFree(dy);

}