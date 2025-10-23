#include <iostream>
#include <omp.h>
#include <vector>
#include "matrixFormat.hpp"
#include "cuda_fp16.h"
// int ceil(int a, int b) {
//     return (a + b - 1) / b;
// }
template <typename T, typename Y>
void csrToBcsr(CSRFormat<T, Y>* csr, BCSRFormat<T, Y>* bcsr) {
    size_t numColRegions = (csr->col + bcsr->TILE_K - 1)/bcsr->TILE_K;
    size_t numRowRegions = (csr->row + bcsr->TILE_M - 1)/bcsr->TILE_M;
    // printf("numColRegions: %d numRowRegions:%d\n", numColRegions, numRowRegions);
    bcsr->blockNum = numColRegions * numRowRegions;
    bcsr->colRegions = numColRegions;
    bcsr->rowRegions = numRowRegions;
    printf("blockNum: %d\n",  bcsr->blockNum);
    bcsr->row = csr->row;
    bcsr->col = csr->col;

    Y* blockInfo_host = (Y *) malloc(sizeof(Y)*  bcsr->blockNum );
    memset(blockInfo_host, 0, sizeof(Y)*  bcsr->blockNum );
    Y sparseBlocks = 0;
    Y nonzeroBlocks = 0;
    printf("csr->row: %d\n", csr->row);
    
    for (int64_t row = 0; row < csr->row; row++) {
        for (int64_t j = csr->rowPtr[row]; j < csr->rowPtr[row+1]; j++) {
            int64_t col = csr->colIdx[j];
            int64_t rowRegion = row/bcsr->TILE_M;
            int64_t colRegion = col/bcsr->TILE_K;
            int64_t blockIndex = rowRegion * numColRegions + colRegion;
            if (blockIndex < 0 ||  blockIndex >= bcsr->blockNum) {
                printf("%ld\n", blockIndex);
            }
            if (blockInfo_host[blockIndex] == 0) {
                nonzeroBlocks += 1;
                blockInfo_host[blockIndex] = 1;
                sparseBlocks++;
            }
        }
    }
    printf("nonzeroBlocks: %d\n", nonzeroBlocks);
    #ifdef DEBUG
    for (int i =0 ;i < 2; i++) {
        printf("blockInfo_host %d\n", blockInfo_host[i]);
    }
    printf("\n");
    printf("sparseBlocks: %d\n", sparseBlocks);
    #endif
    Y relativeIndex = 0;
    Y* relativeBlockIndexMapping_host = (Y*) malloc(bcsr->blockNum * sizeof(Y));
    memset(relativeBlockIndexMapping_host, 0, sizeof(Y)*  bcsr->blockNum );

    for (int64_t i = 0; i < bcsr->blockNum; i++) {
        relativeBlockIndexMapping_host[i] = (blockInfo_host[i] != 0) ? relativeIndex++:Y(-1);
        // printf("relativeBlockIndexMapping_host i:%d %d\n", i, relativeBlockIndexMapping_host[i]);
    }


    Y* bcsrRowPtr_host = (Y*)malloc(sizeof(Y)*(numRowRegions + 1));
    memset(bcsrRowPtr_host, 0, sizeof(Y)* (numRowRegions + 1) );

    Y* bcsrColIdx_host = (Y*)malloc(nonzeroBlocks * sizeof(Y));
    T* bcsrVal_host = (T*)malloc(sizeof(T)* nonzeroBlocks * bcsr->TILE_M  * bcsr->TILE_K);
    memset(bcsrVal_host, 0, sizeof(T)* nonzeroBlocks * bcsr->TILE_M  * bcsr->TILE_K );

    size_t num_blocks = 0;

    for (size_t row = 0; row < csr->row; row += bcsr->TILE_M) {
        bcsrRowPtr_host[row/bcsr->TILE_M] = num_blocks;
        for (size_t col = 0; col < csr->col; col += bcsr->TILE_K) {
            size_t current_block = row / bcsr->TILE_M * numColRegions + col/bcsr->TILE_K;
            if (!blockInfo_host[current_block]) continue;
            bcsrColIdx_host[num_blocks] = col/bcsr->TILE_K;
            num_blocks++;
            // #ifdef DEBUG

            // printf("current_block:%d num_blocks %d col:%d\n", current_block, num_blocks, col);

            // #endif
        }
        #ifdef DEBUG
        printf("bcsrRowPtr_host %d\n", bcsrRowPtr_host[row/bcsr->TILE_M]);
        #endif
    }

    bcsrRowPtr_host[numRowRegions] = num_blocks;

    for (size_t row = 0; row < csr->row; row++)
    {
        for (size_t j = csr->rowPtr[row]; j <  csr->rowPtr[row + 1]; j++) 
        {
            size_t col = csr->colIdx[j];
            // printf("col %d\n", col);
            size_t rowRegion = row / bcsr->TILE_M;
            size_t colRegion = col / bcsr->TILE_K;
            // printf("row_reg %d  col reg %d \n", rowRegion, colRegion);
            size_t blockIndex = rowRegion * numColRegions + colRegion;
            T val = csr->values[j];
            // std::cout << val << std::endl;
            size_t offset = row % bcsr->TILE_M *  bcsr->TILE_K + col %  bcsr->TILE_K;
            size_t bcsrIndex = relativeBlockIndexMapping_host[blockIndex] * bcsr->TILE_M *  bcsr->TILE_K + offset;
            // printf("bcsrIndex:%d\n", bcsrIndex);
            bcsrVal_host[bcsrIndex] = val;
            // printf("blockIndex:%d bcsrVal_host %d offset:%d %f ",blockIndex, bcsrIndex, offset, float( bcsrVal_host[bcsrIndex] ));

        }
    }
    bcsr->values = bcsrVal_host;

    // return;
    bcsr->relativeBlockIndexMapping = relativeBlockIndexMapping_host;

    bcsr->bcsrcolIdx = bcsrColIdx_host;
    bcsr->nonzeroBlocks = nonzeroBlocks;

    bcsr->elemCount =  nonzeroBlocks * bcsr->TILE_M  * bcsr->TILE_K;


    bcsr->bcsrRowPtr = bcsrRowPtr_host;
    return;

}