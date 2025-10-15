#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <stdint.h>
#include <iostream>
#include "ptx.h"
#include "matrixFormat.hpp"
#include "../utils/utils.hpp"
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32
using namespace std;
inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
// __global__ void mmaSmatKernelfp16(half* bcsrValuesA, int* bcsrRowPtrA, int* blockIndexMap , int* bcsrColIdxA, half* B, half* C, int M, int N, int K) {
__global__ void mmaSmatKernelfp16(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B, half *C, size_t M, size_t N, size_t K, int* relativeBlockIndexMapping) {
        //mmaCBTKernel
        const size_t K_tiles = div_ceil(K, MMA_K);
    
        const size_t warp_row = blockIdx.y * MMA_M;
        const size_t warp_col = blockIdx.x * MMA_N;
    
        size_t blockRow = blockIdx.y;
        size_t blockCol = blockIdx.x;
        // printf("blockRow:%d blockCol:%d\n",blockRow, blockCol);
        size_t colRegions = (K + MMA_K - 1) / (MMA_K);
    
        if (warp_row >= M || warp_col >= N) {
            return;
        }
        // printf("blockRow:%d %d %d\n ",blockRow, bcsrRowPtrA[blockRow], bcsrRowPtrA[blockRow+1]);
        // printf("|\n");
        __shared__ half A_smem[MMA_M][MMA_K];
        __shared__ half B_smem[MMA_N][MMA_K];
        __shared__ half C_smem[MMA_M][MMA_N];
    
        const size_t lane_id = threadIdx.x % WARP_SIZE;
        auto group = cooperative_groups::this_thread_block();
    
        uint32_t RC[2] = {0, 0};
    #pragma unroll
        for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow + 1]; ptr++) {
            size_t i = bcsrColIdxA[ptr] / MMA_K;
            // skip empty block
            size_t blockIndex = blockRow * colRegions + i;
            // printf("blockIndex %d\n", blockRow);
            int relativeIndex = relativeBlockIndexMapping[blockIndex];
            // printf("relativeIndex %d\n",relativeIndex );
            // if (relativeIndex == -1) continue;
            size_t A_size = MMA_M * MMA_K * sizeof(half);
            size_t B_size = MMA_N * MMA_K * sizeof(half);
    
             cooperative_groups::memcpy_async(group, &A_smem[0][0],
                             &bcsrValuesA[relativeIndex * MMA_M * MMA_K],
                             A_size);
             cooperative_groups::memcpy_async(group, &B_smem[0][0],
                             &B[i * MMA_K + warp_col * K],
                             B_size);
            
            cooperative_groups::wait(group); // Wait for all copies to complete
            group.sync();
    
            uint32_t RA[4];
            uint32_t RB[2];
    
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);
    
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
            LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);
    
            HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
    
            group.sync();
        }
    
        *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
        *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];
    
        __syncthreads();
    
        if (lane_id < MMA_M) {
            *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
        }
    }
    
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void smatSpmmfp16(int *rowPtr, int *colIdx, int row, int col, int n, int nnz, float* values, float* outvalues) {
    CSRFormat<half, int32_t>csr;
    csr.rowPtr = rowPtr;
    csr.colIdx = colIdx;
    csr.row = row;
    csr.col = col;
    csr.values = (half*)malloc(nnz * sizeof(half));
    for (int i = 0; i < nnz; i++) {
        csr.values[i] = half(values[i]);
    }
    cudaSetDevice(1); 

    BCSRFormat<half, int32_t>bcsr;
    bcsr.TILE_M = 16;
    bcsr.TILE_N = 8;
    bcsr.TILE_K = 16;
    bcsr.nnz = nnz;
    csrToBcsr<half, int32_t>(&csr, &bcsr);
    // return;
    // for (int i = 0; i < 1; i++) {
    //     printf("bcsr.values %d %f\n", i,  float(bcsr.values[i]));
    // }
    //     for (int j = 0; j < 16; j++) {
    //         printf("bcsr.values[%d][%d] = %f\n", i, j, float(bcsr.values[i * bcsr.TILE_M + j]));
    //     }
    //     printf("\n");
    // }
    // printf("bcsr.blockNum = %d\n", bcsr.blockNum);
    // for (int i = 0; i < 2; i++) {
    //     printf("bcsr.blockPtr[%d] = %d\n", i, bcsr.bcsrRowPtr[i]);
    // }
    int m = bcsr.row;
    int k = bcsr.col;
    // printf("m = %d k = %d n = %d\n", m, k, n);
    half* hB = (half*)malloc(k * n * sizeof(half));
    half* hC = (half*)malloc(m * n * sizeof(half));
    half* dC, *dB;
    cudaMalloc((void**)&dC, m * n * sizeof(half));
    cudaMalloc((void**)&dB, k * n * sizeof(half));
    for (int i = 0; i < k * n; i++) {
        hB[i] = 1;
    }
    for (int i = 0; i < m * n; i++) {
        hC[i] = 0;
    }

    cudaMemcpy(dB, hB, k * n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, m * n * sizeof(half), cudaMemcpyHostToDevice);
    // printf("elemCount = %d\n",bcsr.elemCount );
    half* dvalues;
    cudaMalloc(&dvalues, bcsr.elemCount * sizeof(half));
    cudaMemcpy(dvalues, bcsr.values, bcsr.elemCount * sizeof(half), cudaMemcpyHostToDevice);

    // for (int i = 0; i < 16; i++) {
    //     printf("dvalues[%d] = %f\n", i, float( bcsr.values[i]));
    // }
    int *dbcsrRowPtr;
    #ifdef DBUGE
    printf("rowRegions:  %d colRegions %d\n", bcsr.rowRegions, bcsr.colRegions);
    #endif
    cudaMalloc(&dbcsrRowPtr, (bcsr.rowRegions+1) * sizeof(int));

    cudaMemcpy(dbcsrRowPtr, bcsr.bcsrRowPtr, (bcsr.rowRegions+1) * sizeof(int), cudaMemcpyHostToDevice);
    // for (int i = 0; i < 3; i++) {
    //     printf("%d ", bcsr.bcsrRowPtr[i]);
    // }
    // printf("\n");


    int *drelativeBlockIndexMapping;
    cudaMalloc(&drelativeBlockIndexMapping, bcsr.blockNum * sizeof(int));
    cudaMemcpy(drelativeBlockIndexMapping, bcsr.relativeBlockIndexMapping, bcsr.blockNum * sizeof(int), cudaMemcpyHostToDevice);
    int *dbcsrcolIdx;
    cudaMalloc(&dbcsrcolIdx, bcsr.nonzeroBlocks * sizeof(int));

    cudaMemcpy(dbcsrcolIdx, bcsr.bcsrcolIdx, bcsr.nonzeroBlocks * sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid(div_ceil(n, MMA_N), div_ceil(m, MMA_M));
    dim3 block(WARP_SIZE);
    // printf("grid.x %d, grid.y %d\n", grid.x, grid.y);
    auto start = std::chrono::high_resolution_clock::now(); 
    mmaSmatKernelfp16<<<grid, block>>>(dvalues, dbcsrRowPtr,dbcsrcolIdx, dB, dC, m, n, k, drelativeBlockIndexMapping);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now(); 

    auto elapsed = end - start;
    std::cout << "smat 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
   
    CHECK_CUDA_ERROR(cudaGetLastError());

    cudaMemcpy(hC, dC, m * n * sizeof(half), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", float(hC[i*n + j]));
    //         // outvalues[i*n + j] = 1;
    //         // hC[i*n + j] = 1;
    //     }
    //     printf("\n");
    // }
}