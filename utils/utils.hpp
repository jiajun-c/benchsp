#include <iostream>
#include <vector>
#include "matrixFormat.hpp"
#include "cuda_fp16.h"
template <typename T>
void csrToBcsr(CSRFormat<T>* csr, BCSRFormat<T>* bcsr) {
    size_t numColRegions = (csr->col + bcsr->TILE_K - 1)/bcsr->TILE_K;
    size_t numRowRegions = (csr->row + bcsr->TILE_M - 1)/bcsr->TILE_M;
    // printf("numColRegions: %d numRowRegions:%d\n", numColRegions, numRowRegions);
    bcsr->blockNum = numColRegions * numRowRegions;
    bcsr->colRegions = numColRegions;
    bcsr->rowRegions = numRowRegions;
    // printf("blockNum: %d\n",  bcsr->blockNum);
    bcsr->row = csr->row;
    bcsr->col = csr->col;
    int* blockInfo_host = (int *) malloc(sizeof(int)*  bcsr->blockNum );
    memset(blockInfo_host, 0, sizeof(int)*  bcsr->blockNum );
    int sparseBlocks = 0;
    int nonzeroBlocks = 0;

    // printf("csr->row: %d\n", csr->row);
    for (size_t row = 0; row < csr->row; row++) {
        for (size_t j = csr->rowPtr[row]; j < csr->rowPtr[row+1]; j++) {
            size_t col = csr->colIdx[j];
            size_t rowRegion = row/bcsr->TILE_M;
            size_t colRegion = col/bcsr->TILE_K;
            size_t blockIndex = rowRegion * numColRegions + colRegion;
            if (blockInfo_host[blockIndex] == 0) {
                nonzeroBlocks += 1;
                blockInfo_host[blockIndex] = 1;
                sparseBlocks++;
            }
        }
    }
    #ifdef DEBUG
    for (int i =0 ;i < 2; i++) {
        printf("blockInfo_host %d\n", blockInfo_host[i]);
    }
    printf("\n");
    printf("sparseBlocks: %d\n", sparseBlocks);
    #endif
    int relativeIndex = 0;
    int* relativeBlockIndexMapping_host = (int*) malloc(bcsr->blockNum * sizeof(int));
    memset(relativeBlockIndexMapping_host, 0, sizeof(int)*  bcsr->blockNum );

    for (size_t i = 0; i < bcsr->blockNum; i++) {
        relativeBlockIndexMapping_host[i] = (blockInfo_host[i] != 0) ? relativeIndex++:int(-1);
        // printf("relativeBlockIndexMapping_host %d\n", relativeBlockIndexMapping_host[i]);
    }
    int* bcsrRowPtr_host = (int*)malloc(sizeof(int)*(csr->row / bcsr->TILE_M + 1));
    memset(bcsrRowPtr_host, 0, sizeof(int)* (csr->row / bcsr->TILE_M + 1) );

    int* bcsrColIdx_host = (int*)malloc(nonzeroBlocks * sizeof(int));
    T* bcsrVal_host = (T*)malloc(sizeof(T)* nonzeroBlocks * bcsr->TILE_M  * bcsr->TILE_K);
    memset(bcsrVal_host, 0, sizeof(T)* nonzeroBlocks * bcsr->TILE_M  * bcsr->TILE_K );

    size_t num_blocks = 0;

    for (size_t row = 0; row < csr->row; row += bcsr->TILE_M) {
        bcsrRowPtr_host[row/bcsr->TILE_M] = num_blocks;
        for (size_t col = 0; col < csr->col; col += bcsr->TILE_K) {
            size_t current_block = row / bcsr->TILE_M * numColRegions + col/bcsr->TILE_K;
            if (!blockInfo_host[current_block]) continue;
            bcsrColIdx_host[num_blocks] = col;
            num_blocks++;
            // #ifdef DEBUG

            // printf("current_block:%d num_blocks %d col:%d\n", current_block, num_blocks, col);

            // #endif
        }
        #ifdef DEBUG
        printf("bcsrRowPtr_host %d\n", bcsrRowPtr_host[row/bcsr->TILE_M]);
        #endif
    }
    bcsrRowPtr_host[csr->row/bcsr->TILE_M+1] = num_blocks;

    for (size_t row = 0; row < csr->row; row++)
    {
        // printf("%d %d\n",  csr->rowPtr[row],  csr->rowPtr[row+1]);
        for (size_t j = csr->rowPtr[row]; j <  csr->rowPtr[row + 1]; j++) 
        {
            size_t col = csr->colIdx[j];
            // printf("col %d\n", col);
            size_t rowRegion = row / bcsr->TILE_M;
            size_t colRegion = col / bcsr->TILE_K;
            // printf("row_reg %d  col reg %d \n", row, col);
            size_t blockIndex = rowRegion * numColRegions + colRegion;
            T val = csr->values[j];
            // std::cout << val << std::endl;
            size_t offset = row % bcsr->TILE_M *  bcsr->TILE_K + col %  bcsr->TILE_K;
            size_t bcsrIndex = relativeBlockIndexMapping_host[blockIndex] * bcsr->TILE_M *  bcsr->TILE_K + offset;
            bcsrVal_host[bcsrIndex] = val;
            // printf("blockIndex:%d bcsrVal_host %d offset:%d %f ",blockIndex, bcsrIndex, offset, float( bcsrVal_host[bcsrIndex] ));

        }
    }
    bcsr->values = bcsrVal_host;
    // printf("bcsr->values %f ", float(bcsrVal_host[15]));
    bcsr->relativeBlockIndexMapping = relativeBlockIndexMapping_host;
    bcsr->bcsrRowPtr = bcsrRowPtr_host;
    bcsr->bcsrcolIdx = bcsrColIdx_host;
    bcsr->nonzeroBlocks = nonzeroBlocks;
    bcsr->elemCount =  nonzeroBlocks * bcsr->TILE_M  * bcsr->TILE_K;
    return;
}