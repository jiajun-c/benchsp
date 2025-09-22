#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <stdint.h>
#include <iostream>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32
using namespace std;
inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
__global__ void mmaSmatKernelfp16(half* bcsrValuesA, int* bcsrRowPtrA, int* blockIndexMap , int* bcsrColIdxA, half* B, half* C, int M, int N, int K) {
    const size_t k_tiles = div_ceil(K, MMA_K);
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    size_t blockRow = blockIdx.y;
    size_t blockCol = blockIdx.x;
    size_t colRegions = (K + MMA_K - 1)/MMA_K;
    if (warp_row >= M || warp_row >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;
    auto group = cooperative_groups::this_thread_block();
    uint32_t RC[2] = {0, 0};
    #pragma unroll
    for (size_t ptr = bcsrRowPtrA[blockRow]; ptr < bcsrRowPtrA[blockRow+1]; ptr++) {
        size_t i = bcsrColIdxA[ptr]/MMA_K; // 位于这个一行的第几个块上
        size_t blockIndex = blockRow * colRegions + i; // 块的实际位置
        size_t relativeIndex = blockIndexMap[blockIndex];
        size_t A_size = MMA_M * MMA_K * sizeof(half); // A 块的字节大小
        size_t B_size = MMA_N * MMA_K * sizeof(half); // B 块的字节大小

        cooperative_groups::memcpy_async(group, &A_smem[0][0], &bcsrValuesA[relativeIndex * MMA_M * MMA_K], A_size);
        cooperative_groups::memcpy_async(group, &B_smem[0][0], &B[i*MMA_K + warp_col * K], B_size);
        cooperative_groups::wait(group);
        group.sync();

        uint32_t RA[4], RB[2];
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[land_id % 16][(lane_id / 16) * 8]);
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

