#include "common.h"
#include "matrixFormat.hpp"
#define BLOCK_INFO
using namespace std;

#define setbit(x, y) x |= (1 << y)    // set the yth bit of x is 1
#define clrbit(x, y) x &= ~(1 << y)   // set the yth bit of x is 0
#define getbit(x, y) ((x) >> (y) & 1) // get the yth bit of x

void csrTodbsr(CSRFormat<double, int32_t> *csr, dbsrMat *bsr) {
    int row = csr->row;
    int col = csr->col;
    int block_len = 4;
    int brow = (row+3)/4 ;
    int *rowPtr = (int *)malloc(sizeof(int) * (brow+1));
    memset(rowPtr, 0, sizeof(int) * ((row+3)/4 + 1));
    for (int i = 0; i < row; i++) {
        int start = csr->rowPtr[i];
        int end = csr->rowPtr[i + 1];
        int len = end - start;
        rowPtr[i/block_len + 1] = max(rowPtr[i/block_len + 1], (len + block_len - 1)/block_len);
    }
    for (int i = 1; i <= brow; i++) {
        rowPtr[i] += rowPtr[i-1];
    }
    int total_block = rowPtr[brow];
#ifdef BLOCK_INFO
    printf("row:%d col:%d nnz:%d brow:%d total_block:%d \n", row, col, csr->nnz, brow, total_block);
#endif
    printf("total_block: %d nnz:%d \n", total_block, csr->nnz);
    int *colIdx = (int *)malloc(block_len * block_len * sizeof(double) * total_block);
    memset(colIdx, 0, block_len * block_len * sizeof(double) * total_block);
    unsigned short *bmap = (unsigned short *)malloc(sizeof(unsigned short) * total_block);
    double *blcVal = (double *)malloc(block_len * block_len * sizeof(double) * total_block);
    memset(blcVal, 0, block_len * block_len * sizeof(double) * total_block);

    for (int i = 0; i < row; i++) {
        int start = csr->rowPtr[i];
        int end = csr->rowPtr[i + 1];
        int z = 0;
        int iblock = i/block_len;
        int rowinb = i%block_len;
        for (int j = start; j < end; j++, z++) {
            int jblock = z/ block_len;
            int colinb = z%block_len;
            int targetblock = rowPtr[iblock] + jblock;
            // printf("%d\n", targetblock);
            blcVal[targetblock * block_len * block_len + rowinb * block_len + colinb] = csr->values[j];
            colIdx[targetblock * block_len * block_len + rowinb * block_len + colinb] = csr->colIdx[j];
            setbit(bmap[targetblock ], (rowinb * block_len + colinb));
        }
    }

    bsr->colIdx = colIdx;
    bsr->blcVal = blcVal;
    bsr->blcPtr = rowPtr;
    bsr->row = row;
    bsr->col = col;
    bsr->nnz = csr->nnz;
    bsr->blc_row = brow;
    bsr->blcMap = bmap;
    bsr->blocknnz = total_block*block_len*block_len;
    // return ;

}