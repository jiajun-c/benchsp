#pragma once
#include <iostream>
#include <vector>

template <typename T, typename Y>
class CSRFormat {
public:
    Y* rowPtr;
    Y* colIdx;
    T* values;
    Y row, col;
    Y nnz;
};

template <typename T, typename Y>
class BCSRFormat {
public:
    Y* bcsrRowPtr;
    Y* bcsrcolIdx;
    Y *relativeBlockIndexMapping;
    T* values;
    Y row, col, blockNum;
    int TILE_M;
    int TILE_N;
    int TILE_K;
    Y nnz;
    Y colRegions, rowRegions, nonzeroBlocks, elemCount;
    Y block_row;
    Y block_col;
};

// template <typename T, typename Y>
// class bsrMAT
// {
//     public:
//         int row;
//         int col;
//         int nnz;
//         int blc_row;
//         int blc_col;
//         int blc_num;
//         double stand;
//         double avg_nnz;
//         T *blcPtr;
//         T *blcIdx;
//         Y *blcVal;
//         unsigned short *blcMap;
//         T warpnum;       // load-balanced value
//         T *rowPtrbyWarp; // load-balanced array
//         T *rowIdxbyWarp; // load-balanced array
//         float *blcVal_fp32;         // float
//         float *dVecX_fp32;
//         float *dVecY_fp32;
//         uint32_t *blcVal_fp16; // half
//         uint32_t *dVecX_fp16;
//         uint32_t *dVecY_fp16;
// } ;

class bsrMAT
{
    public:
        int row;
        int col;
        int nnz;
        int blc_row;
        int blc_col;
        int blc_num;
        double stand;
        double avg_nnz;
        int32_t *blcPtr;
        int32_t *blcIdx;
        double *blcVal;
        unsigned short *blcMap;
        int32_t warpnum;       // load-balanced value
        int32_t *rowPtrbyWarp; // load-balanced array
        int32_t *rowIdxbyWarp; // load-balanced array
        float *blcVal_fp32;         // float
        float *dVecX_fp32;
        float *dVecY_fp32;
        uint32_t *blcVal_fp16; // half
        uint32_t *dVecX_fp16;
        uint32_t *dVecY_fp16;
} ;


class dbsrMat {
    public:
        int row;
        int col;
        int nnz;
        int blc_row;
        int blc_col;
        int blc_num;
        double stand;
        double avg_nnz;
        int32_t *blcPtr;
        int32_t *colIdx;
        double *blcVal;
        unsigned short *blcMap;
        int32_t blocknnz;
        int32_t warpnum;       // load-balanced value
        int32_t *rowPtrbyWarp; // load-balanced array
        int32_t *rowIdxbyWarp; // load-balanced array
};