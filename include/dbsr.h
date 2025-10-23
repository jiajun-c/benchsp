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
    int *rowPtrSorted = (int *)malloc(sizeof(int) * (brow+1));
    memset(rowPtrSorted, 0, sizeof(int) * ((row+3)/4 + 1));
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
    int* lens = (int *)malloc(sizeof(int) * row);
    for (int i = 0; i < row; i++) {
        int start = csr->rowPtr[i];
        int end = csr->rowPtr[i + 1];
        int len = end - start;
        lens[i] = len;
    }
    sort(lens, lens+row);
    // for (int i = 0; i < row; i++) {
    //     int len = lens[i];
    //     rowPtrSorted[i/block_len + 1] = max(rowPtrSorted[i/block_len + 1], (len + block_len - 1)/block_len);
    // }
    // for (int i = 1; i <= brow; i++) {
    //     rowPtrSorted[i] += rowPtrSorted[i-1];
    // }
    int total_block_sorted = rowPtrSorted[brow];
#ifdef BLOCK_INFO
    double avg_nnz = (double)csr->nnz/ total_block;
    printf("====denser block info====\n");
    printf("row:%d col:%d nnz:%d brow:%d total_block:%d \n", row, col, csr->nnz, brow, total_block);
    printf("avglen: %f\n", (double)csr->nnz/row);
    printf("avg_nnz: %f\n", avg_nnz);
    printf("avg block in block row: %f\n", total_block/(double)brow);
    printf("====denser block info end====\n\n");

    // printf("====denser block with resort info====\n");
    // printf("total_block_sorted: %d\n", total_block_sorted);
    // printf("avg_nnz_sorted: %f\n", double(csr->nnz)/total_block_sorted);
#endif
    int *colIdx = (int *)malloc(block_len * block_len * sizeof(double) * total_block);
    memset(colIdx, 0, block_len * block_len * sizeof(double) * total_block);
    unsigned short *bmap = (unsigned short *)malloc(sizeof(unsigned short) * total_block);
    memset(bmap, 0, sizeof(unsigned short) * total_block);
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
            // printf("i:%d j:%d z:%d iblock:%d jblock:%d rowinb:%d colinb:%d\n", i, j, z, iblock, jblock, rowinb, colinb);
            int targetblock = rowPtr[iblock] + jblock;
            blcVal[targetblock * block_len * block_len + rowinb * block_len + colinb] = csr->values[j];
            colIdx[targetblock * block_len * block_len + rowinb * block_len + colinb] = csr->colIdx[j];
            setbit(bmap[targetblock], (rowinb * block_len + colinb));
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
void balanceDbsr(dbsrMat *dbsr) { 
    int row = dbsr->row;
    int col = dbsr->col;
    int nnz = dbsr->nnz;
    printf("====balance denser block info====\n");
    printf("row:%d col:%d nnz:%d brow:%d total_block:%d \n", row, col, nnz, dbsr->blc_row, dbsr->blocknnz);
    int *rowPtrbyWarp = (int *)malloc(sizeof(int) * (dbsr->blc_row+1));
    vector<int>row_info(dbsr->blc_row+1);
    int now = 0;
    int warpcount = 0;
        // printf("hello\n");

    for (int i = 0; i < dbsr->blc_row; i++) {
        // return;
        // printf("i:%d len:%d\n", i, (dbsr->blcPtr[i+1] - dbsr->blcPtr[i]));
        int len = (dbsr->blcPtr[i+1] - dbsr->blcPtr[i]);
        if (now + len <= WARP_CAPACITY) {
            row_info[i] = 0;
            now += len;
        } else {
            if (now != 0) {
                warpcount++;
                row_info[i] = 1;
                now = 0;
            }
            if (len <= WARP_CAPACITY) {
                now = len;
            } else {
                warpcount += (len+WARP_CAPACITY-1)/WARP_CAPACITY;
                row_info[i] = (len+WARP_CAPACITY-1)/WARP_CAPACITY;
            }
        }
    }
    // 如果还有剩余的，添加一个warp进行处理
    if (now) {
        // printf("now:%d\n", now);
        warpcount++;
    }

    // 一共使用warpcount个warp
    int now_warp = 0;
    vector<vector<int>>process_row(warpcount, vector<int>());
    for (int i = 0; i < dbsr->blc_row; i++) {
        int len = (dbsr->blcPtr[i+1] - dbsr->blcPtr[i]);
        // printf("i:%d len:%d\n", i, len);
        if (now + len <= WARP_CAPACITY) {
            process_row[now_warp].push_back(i);
            now += len;
        } else {
            if (now != 0) {
                now = 0;
            }
            if (len <= WARP_CAPACITY) {
                now_warp++;
                process_row[now_warp].push_back(i);
            } else {
                for (int j = 1; j <= (len + WARP_CAPACITY - 1)/WARP_CAPACITY; j++) {
                    now_warp++;
                    process_row[now_warp].push_back(i);
                }
            }
        }
    }
    int* prefix = (int *)malloc(sizeof(int) * (warpcount+1));
    memset(prefix, 0, sizeof(int) * (warpcount+1));
    // vector<int>prefix(warpcount+1);
    for (int i = 0; i < warpcount; i++) {
        prefix[i+1] = prefix[i] + process_row[i].size();
    }
    // for (int i = 0; i < 100; i++) {
    //     printf("prefix:%d = %d\n", i, dbsr->prefix[i]);
    // }
    dbsr->prefix = prefix;
    dbsr->warpnum = warpcount;
    printf("warpcount:%d\n", warpcount);
}
void csrRecorderSelf(CSRFormat<double, int32_t> *csr,CSRFormat<double, int32_t> *recsr) {
    vector<pair<int, int>> rowInfo;
    int row = csr->row;
    int col = csr->col;
    int nnz = csr->nnz;
    for (int i = 0; i < csr->row; i++) {
        rowInfo.push_back(make_pair(csr->rowPtr[i+1]-csr->rowPtr[i], i));
    }
    sort(rowInfo.begin(), rowInfo.end());
    int* rowPtrReorder = (int *)malloc(sizeof(int) * (row+1));
    int *colIdxRecorder = (int *)malloc(sizeof(int) * nnz);
    // 通过recorder map将数据重排回去
    int *reorderMap = (int *)malloc(sizeof(int) * row);
    double *valuesRecorder = (double *)malloc(sizeof(double) * nnz);
    rowPtrReorder[0] = 0;
    for (int i = 0; i < row; i++) {
        int oldIndex = rowInfo[i].second;
        rowPtrReorder[i+1] = csr->rowPtr[oldIndex+1] - csr->rowPtr[oldIndex];
        reorderMap[i] = oldIndex;
    }
    for (int i = 1; i <= row; i++) {
        rowPtrReorder[i] += rowPtrReorder[i-1];
    }
    for (int i = 0; i < row; i++) {
        int oldIndex = rowInfo[i].second;
        for (int j = 0; j < rowPtrReorder[i+1]-rowPtrReorder[i]; j++) {
            colIdxRecorder[rowPtrReorder[i]+j] = csr->colIdx[csr->rowPtr[oldIndex]+j];
            valuesRecorder[rowPtrReorder[i]+j] = csr->values[csr->rowPtr[oldIndex]+j];
        }
    }
    recsr->col = col;
    recsr->row = row;
    recsr->nnz = nnz;
    recsr->colIdx = colIdxRecorder;
    recsr->values = valuesRecorder;
    recsr->rowPtr = rowPtrReorder;
}
    // return;
void csrTodbsrWithRecorder(CSRFormat<double, int32_t> *csr, dbsrMat *bsr) {
    vector<pair<int, int>> rowInfo;
    int row = csr->row;
    int col = csr->col;
    int nnz = csr->nnz;
    for (int i = 0; i < csr->row; i++) {
        rowInfo.push_back(make_pair(csr->rowPtr[i+1]-csr->rowPtr[i], i));
    }
    sort(rowInfo.begin(), rowInfo.end());
    int* rowPtrReorder = (int *)malloc(sizeof(int) * (row+1));
    int *colIdxRecorder = (int *)malloc(sizeof(int) * nnz);
    // 通过recorder map将数据重排回去
    int *reorderMap = (int *)malloc(sizeof(int) * row);
    double *valuesRecorder = (double *)malloc(sizeof(double) * nnz);
    rowPtrReorder[0] = 0;
    for (int i = 0; i < row; i++) {
        int oldIndex = rowInfo[i].second;
        rowPtrReorder[i+1] = csr->rowPtr[oldIndex+1] - csr->rowPtr[oldIndex];
        reorderMap[i] = oldIndex;
    }
    for (int i = 1; i <= row; i++) {
        rowPtrReorder[i] += rowPtrReorder[i-1];
    }
    for (int i = 0; i < row; i++) {
        int oldIndex = rowInfo[i].second;
        for (int j = 0; j < rowPtrReorder[i+1]-rowPtrReorder[i]; j++) {
            colIdxRecorder[rowPtrReorder[i]+j] = csr->colIdx[csr->rowPtr[oldIndex]+j];
            valuesRecorder[rowPtrReorder[i]+j] = csr->values[csr->rowPtr[oldIndex]+j];
        }
    }
    // return;

    CSRFormat<double, int32_t> csr_reorder;
    csr_reorder.row = row;
    csr_reorder.col = col;
    csr_reorder.nnz = nnz;
    csr_reorder.rowPtr = rowPtrReorder;
    csr_reorder.colIdx = colIdxRecorder;
    csr_reorder.values = valuesRecorder;
    bsr->recorderMap = reorderMap;
    // for (int i = 0; i < row; i++) {
    //     printf("%d\n", reorderMap[i]);
    // }
    csrTodbsr(&csr_reorder, bsr);
    balanceDbsr(bsr);
}

