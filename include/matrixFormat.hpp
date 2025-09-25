#pragma once
#include <iostream>
#include <vector>

template <typename T>
class CSRFormat {
public:
    int* rowPtr;
    int* colIdx;
    T* values;
    int row, col;
    int nnz;
};

template <typename T>
class BCSRFormat {
public:
    int* bcsrRowPtr;
    int* bcsrcolIdx;
    int *relativeBlockIndexMapping;
    T* values;
    int row, col, blockNum;
    int TILE_M;
    int TILE_N;
    int TILE_K;
    int nnz;
    int colRegions, rowRegions, nonzeroBlocks, elemCount;
};

