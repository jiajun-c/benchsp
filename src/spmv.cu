#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string.h>
#include <cuda_fp16.h>
#include <chrono>
#include "matrixFormat.hpp"
using namespace std;
#define WARP 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK 128  // WARP_SIZE * WARPS_PER_BLOCK

__global__ void spmv_fp16(int *rowPtr, int *colIdx, int *shortMap, int rows, int nnz, half* values,half* x, half* outvalues) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;
    // if (tid == 0) {
    //     printf("short rows: %d %d mapID:%d values:%f xval:%f\n", rowPtr[tid], rowPtr[tid+1], shortMap[tid], float( values[0]));
    // }
    half tmp = 0.0;
    for (int i = rowPtr[tid]; i < rowPtr[tid+1]; i++) {
        int col = colIdx[i];
        half val = values[i];
        half xval = x[col];
        tmp += __half2float(xval) * __half2float(val);
    }
    outvalues[shortMap[tid]] = tmp;
}
__global__ void spmv_fp64(int64_t *rowPtr, int64_t *colIdx, int64_t *shortMap, int64_t rows, int64_t nnz, double* values,double* x, double* outvalues) { 
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows) return;
    // if (tid == 0) {
    //     printf("short rows: %d %d mapID:%d values:%f xval:%f\n", rowPtr[tid], rowPtr[tid+1], shortMap[tid], float( values[0]));
    // }
    double tmp = 0.0;
    for (int64_t i = rowPtr[tid]; i < rowPtr[tid+1]; i++) {
        int64_t col = colIdx[i];
        double val = values[i];
        double xval = x[col];
        tmp += xval * val;
    }
    outvalues[shortMap[tid]] = tmp;
}

__global__ void spmv_fp16_long(int *rowPtr, int *colIdx, int *longMap, int rows, int nnz, half* values,half* x, half* outvalues) { 
    const size_t warp_id = threadIdx.x / WARP;
    const size_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    // if (warp_col == 9696) {
    //     printf("have\n");
    // }
    if (warp_col >= rows) return;
    half tmp = 0.0;
    int count = rowPtr[warp_col+1] - rowPtr[warp_col];
    int repeat = (count + WARP - 1) / WARP;
    int laneID = threadIdx.x % WARP;
    for (int i = 0; i < repeat; i++) {
        int start =  i * WARP + laneID;
        if (start >= count) continue;
        start += rowPtr[warp_col];
        int col = colIdx[start];
        half val = values[start];
        half xval = x[col];
        tmp += __half2float(xval) * __half2float(val);
    } 
    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = WARP / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }
    // if (longMap[warp_col]== 39825) {
    //     printf("now\n");
    // }
    if (laneID == 0) {
        outvalues[longMap[warp_col]] = __float2half(tmp);
    }
}

__global__ void spmv_fp64_long(int64_t *rowPtr, int64_t *colIdx, int64_t *longMap, int64_t rows, int64_t nnz, double* values,double* x, double* outvalues) { 
    const int64_t warp_id = threadIdx.x / WARP;
    const int64_t warp_col = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    // if (warp_col == 9696) {
    //     printf("have\n");
    // }
    if (warp_col >= rows) return;
    double tmp = 0.0;
    int64_t count = rowPtr[warp_col+1] - rowPtr[warp_col];
    int64_t repeat = (count + WARP - 1) / WARP;
    int laneID = threadIdx.x % WARP;
    for (int i = 0; i < repeat; i++) {
        int64_t start =  i * WARP + laneID;
        if (start >= count) continue;
        start += rowPtr[warp_col];
        int col = colIdx[start];
        double val = values[start];
        double xval = x[col];
        tmp += xval * val;
    } 
    constexpr unsigned int mask = 0xffffffff;
#pragma unroll
    for (size_t i = WARP / 2; i >= 1; i /= 2) {
        tmp += __shfl_xor_sync(mask, tmp, i);
    }
    // if (longMap[warp_col]== 39825) {
    //     printf("now\n");
    // }
    if (laneID == 0) {
        outvalues[longMap[warp_col]] = tmp;
    }
}
template <typename T, typename Y>
struct shortRow
{
    T *rowPtr;
    T *colIdx;
    Y *values;
    T rows; // 短行数目
    T nnz;
    T *indexMap;
};

template <typename T, typename Y>
struct longRow
{
    T *rowPtr;
    T *colIdx;
    Y *values;
    T rows;
    T nnz;
    T *indexMap;
};

template <typename T, typename Y>
void preprocess(T *rowPtr, T *colIdx, T rows, T cols, T nnz, Y* values, T threshold, shortRow<T, Y>* shortMat, longRow<T, Y>* longMat) {
    vector<T>shortRows, longRows;
    vector<T>shortColIdx, longColIdx;
    vector<Y>shortValues, longValues;
    vector<T>shortRowPtr, longRowPtr;
    shortRowPtr.push_back(0);
    longRowPtr.push_back(0);
    T shortNow = 0;
    T longNow = 0;
    for (T i = 0; i < rows; i++) {
        if (rowPtr[i+1] - rowPtr[i] > threshold) {
            longRows.push_back(i);
            longNow += rowPtr[i+1] - rowPtr[i];
            longRowPtr.push_back(longNow);

            for (T j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                longValues.push_back(values[j]);
                longColIdx.push_back(colIdx[j]);
            }
        } else {
            shortRows.push_back(i);
            shortNow += rowPtr[i+1] - rowPtr[i];
            shortRowPtr.push_back(shortNow);

            for (T j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                shortValues.push_back(values[j]);
                shortColIdx.push_back(colIdx[j]);
            }
        }
    }
    shortMat->rowPtr = (T *)malloc(shortRowPtr.size() * sizeof(T));
    for (auto i = 0; i < shortRowPtr.size(); i++) shortMat->rowPtr[i] = shortRowPtr[i];
    // printf("shortMap rowPtr 1: %d\n", shortRowPtr[1]);
    shortMat->colIdx = (T *)malloc(shortColIdx.size() * sizeof(T));
    for (auto i = 0; i < shortColIdx.size(); i++) shortMat->colIdx[i] = shortColIdx[i];

    shortMat->values = (Y *)malloc(shortValues.size() * sizeof(Y));
    for (auto i = 0; i < shortValues.size(); i++) shortMat->values[i] = shortValues[i];

    shortMat->indexMap = (T *)malloc(shortRows.size() * sizeof(T));
    for (auto i = 0; i < shortRows.size(); i++) shortMat->indexMap[i] = shortRows[i];

    // shortMat->rowPtr = shortRowPtr.data();
    // shortMat->colIdx = shortColIdx.data();
    // shortMat->values = shortValues.data();
    shortMat->rows = shortRowPtr.size() - 1;
    shortMat->nnz = shortNow;

    longMat->rowPtr = (T *)malloc(longRowPtr.size() * sizeof(T));
    for (auto i = 0; i < longRowPtr.size(); i++) longMat->rowPtr[i] = longRowPtr[i];

    longMat->colIdx = (T *)malloc(longColIdx.size() * sizeof(T));
    for (auto i = 0; i < longColIdx.size(); i++) longMat->colIdx[i] = longColIdx[i];

    longMat->values = (Y *)malloc(longValues.size() * sizeof(Y));
    for (auto i = 0; i < longValues.size(); i++) longMat->values[i] = longValues[i];

    longMat->indexMap = (T *)malloc(longRows.size() * sizeof(T));
    for (auto i = 0; i < longRows.size(); i++) longMat->indexMap[i] = longRows[i];
    // longMat->rowPtr = longRowPtr.data();
    // longMat->colIdx = longColIdx.data();
    // longMat->values = longValues.data();
    longMat->rows = longRowPtr.size() - 1;
    longMat->nnz = longNow;
}
void MySpmvWithPrefp16(int32_t *rowPtr, int32_t *colIdx, int32_t rows, int32_t cols, int32_t nnz, float* values,float* x, float* outvalues) {
    cudaSetDevice(1);
    shortRow<int32_t, half> shortMat;
    longRow<int32_t, half> longMat;
    half* d_values;
    d_values = (half*)malloc(nnz * sizeof(half));
    for (int i = 0; i < nnz; i++) {
        d_values[i] = half(values[i]);
    }
    preprocess<int32_t, half>(rowPtr, colIdx, rows, cols, nnz, d_values, 128, &shortMat, &longMat);
    half* short_d_values,*long_d_values, *d_x, *h_x, *h_y, *d_y;
    h_x = (half*)malloc(cols * sizeof(half));
    h_y = (half*)malloc(rows * sizeof(half));

    for (int i = 0; i < cols; i++) {
        h_x[i] = half(x[i]);
    }
    cudaStream_t my_stream[3];
    cudaStreamCreate(&my_stream[0]);
    cudaStreamCreate(&my_stream[1]);
    cudaMalloc((void**)&short_d_values, shortMat.nnz * sizeof(half));
    cudaMalloc((void**)&long_d_values, longMat.nnz * sizeof(half));

    cudaMalloc((void**)&d_x,      cols * sizeof(half));
    cudaMalloc((void**)&d_y,      rows * sizeof(half));
    // printf("short nnz :%d long nnz :%d\n", shortMat.nnz, longMat.nnz);
    // printf("shortMat.values: %f\n", float(shortMat.values[0]));

    cudaMemcpy(short_d_values, shortMat.values, shortMat.nnz * sizeof(half), cudaMemcpyHostToDevice);
    // return;

    cudaMemcpy(long_d_values, longMat.values,   longMat.nnz * sizeof(half), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x, cols * sizeof(half), cudaMemcpyHostToDevice);
    
    int* short_d_rowPtr, *short_d_colIdx;
    int* long_d_rowPtr, *long_d_colIdx;
    int* longMap, *shortMap;
    cudaMalloc((void**)&short_d_rowPtr, (shortMat.rows+1) * sizeof(int));
    cudaMalloc((void**)&short_d_colIdx, shortMat.nnz * sizeof(int));
    cudaMalloc((void**)&shortMap, shortMat.rows * sizeof(int));
    cudaMalloc((void**)&longMap, longMat.rows * sizeof(int));
    cudaMemcpy(short_d_rowPtr, shortMat.rowPtr, (shortMat.rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(short_d_colIdx, shortMat.colIdx, shortMat.nnz * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&long_d_rowPtr, (longMat.rows+1) * sizeof(int));
    cudaMalloc((void**)&long_d_colIdx, longMat.nnz * sizeof(int));
    cudaMemcpy(long_d_rowPtr, longMat.rowPtr, (longMat.rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(long_d_colIdx, longMat.colIdx, longMat.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(longMap, longMat.indexMap, longMat.rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(shortMap, shortMat.indexMap, shortMat.rows * sizeof(int), cudaMemcpyHostToDevice);

    int thread_size_short = 128; 
    int thread_size_long = 128; 
    // printf("short size: %d\n", thread_size_short);
    int block_size_short = (shortMat.rows + thread_size_short - 1)/thread_size_short;
    int block_size_long = (longMat.rows + thread_size_long - 1)/4;
    printf("longMat.rows %d long size: %d block_size_long:%d \n",longMat.rows, thread_size_long, block_size_long);
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now(); 

        spmv_fp16<<<block_size_short, thread_size_short, 0, my_stream[0]>>>(short_d_rowPtr, short_d_colIdx, shortMap, shortMat.rows, shortMat.nnz, short_d_values, d_x, d_y);
        spmv_fp16_long<<<block_size_long, thread_size_long, 0, my_stream[1]>>>(long_d_rowPtr, long_d_colIdx, longMap, longMat.rows, longMat.nnz, long_d_values, d_x, d_y);
        cudaStreamSynchronize(my_stream[0]);
        cudaStreamSynchronize(my_stream[1]);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now(); 

        auto elapsed = end - start;
        std::cout << "MySPMV 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    }
    cudaMemcpy(h_y, d_y, rows * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; i++) {
        outvalues[i] = float(h_y[i]);
    }
}
void MySpmv(int *rowPtr, int *colIdx, int rows, int cols, int nnz, float* values,float* x, float* outvalues) {
    int thread_size = 256; 
    int block_size = (rows + thread_size - 1)/thread_size;
    int cnt1 = 0, cnt2 = 0;
    for (int i = 0; i < rows; i++) {
        // printf("%d\n", rowPtr[i+1] - rowPtr[i]);
        if (rowPtr[i+1] - rowPtr[i] > 128) cnt1++;
        else cnt2++;
    }
    printf("cnt1:%d cnt2:%d\n", cnt1, cnt2);
    half* d_values, *h_values, *d_x, *h_x, *h_y, *d_y;
    h_values = (half*)malloc(nnz * sizeof(half));
    h_y = (half*)malloc(rows * sizeof(half));
    h_x = (half*)malloc(cols * sizeof(half));

    for (int i = 0; i < nnz; i++) {
        h_values[i] = half(values[i]);
    }
    for (int i = 0; i < cols; i++) {
        h_x[i] = half(x[i]);
    }
    cudaMalloc((void**)&d_values, nnz * sizeof(half));
    cudaMalloc((void**)&d_x,      cols * sizeof(half));
    cudaMalloc((void**)&d_y,      rows * sizeof(half));

    cudaMemcpy(d_values, h_values, nnz * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, cols * sizeof(half), cudaMemcpyHostToDevice);
    
    int* d_rowPtr, *d_colIdx;
    cudaMalloc((void**)&d_rowPtr, (rows+1) * sizeof(int));
    cudaMalloc((void**)&d_colIdx, nnz * sizeof(int));
    cudaMemcpy(d_rowPtr, rowPtr, (rows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, (nnz) * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now(); 

        // spmv_fp16<<<block_size, thread_size>>>(d_rowPtr, d_colIdx, nullptr, nnz, d_values, d_x, d_y);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now(); 

        auto elapsed = end - start;
        std::cout << "MySPMV 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    }
    cudaMemcpy(h_y, d_y, rows * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; i++) {
        outvalues[i] = float(h_y[i]);
    }
}

void MySpmvWithPrefp64(int64_t *rowPtr, int64_t *colIdx, int64_t rows, int64_t cols, int64_t nnz, double* values,double* x, double* outvalues) { 
    shortRow<int64_t, double> shortMat;
    longRow<int64_t, double> longMat;
    double* d_values;
    d_values = (double*)malloc(nnz * sizeof(double));
    for (int i = 0; i < nnz; i++) {
        d_values[i] = double(values[i]);
    }
    preprocess<int64_t, double>(rowPtr, colIdx, rows, cols, nnz, d_values, 128, &shortMat, &longMat);
    double* short_d_values,*long_d_values, *d_x, *h_x, *h_y, *d_y;
    h_x = (double*)malloc(cols * sizeof(double));
    h_y = (double*)malloc(rows * sizeof(double));

    for (int i = 0; i < cols; i++) {
        h_x[i] = double(x[i]);
    }
    cudaStream_t my_stream[3];
    cudaStreamCreate(&my_stream[0]);
    cudaStreamCreate(&my_stream[1]);
    cudaMalloc((void**)&short_d_values, shortMat.nnz * sizeof(double));
    cudaMalloc((void**)&long_d_values, longMat.nnz * sizeof(double));

    cudaMalloc((void**)&d_x,      cols * sizeof(double));
    cudaMalloc((void**)&d_y,      rows * sizeof(double));
    // printf("short nnz :%d long nnz :%d\n", shortMat.nnz, longMat.nnz);
    // printf("shortMat.values: %f\n", float(shortMat.values[0]));

    cudaMemcpy(short_d_values, shortMat.values, shortMat.nnz * sizeof(double), cudaMemcpyHostToDevice);
    // return;

    cudaMemcpy(long_d_values, longMat.values,   longMat.nnz * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x, cols * sizeof(double), cudaMemcpyHostToDevice);
    
    int64_t* short_d_rowPtr, *short_d_colIdx;
    int64_t* long_d_rowPtr, *long_d_colIdx;
    int64_t* longMap, *shortMap;
    cudaMalloc((void**)&short_d_rowPtr, (shortMat.rows+1) * sizeof(int64_t));
    cudaMalloc((void**)&short_d_colIdx, shortMat.nnz * sizeof(int64_t));
    cudaMalloc((void**)&shortMap, shortMat.rows * sizeof(int64_t));
    cudaMalloc((void**)&longMap, longMat.rows * sizeof(int64_t));
    cudaMemcpy(short_d_rowPtr, shortMat.rowPtr, (shortMat.rows+1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(short_d_colIdx, shortMat.colIdx, shortMat.nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&long_d_rowPtr, (longMat.rows+1) * sizeof(int64_t));
    cudaMalloc((void**)&long_d_colIdx, longMat.nnz * sizeof(int64_t));
    cudaMemcpy(long_d_rowPtr, longMat.rowPtr, (longMat.rows+1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(long_d_colIdx, longMat.colIdx, longMat.nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(longMap, longMat.indexMap, longMat.rows * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(shortMap, shortMat.indexMap, shortMat.rows * sizeof(int64_t), cudaMemcpyHostToDevice);

    int thread_size_short = 128; 
    int thread_size_long = 128; 
    // printf("short size: %d\n", thread_size_short);
    int block_size_short = (shortMat.rows + thread_size_short - 1)/thread_size_short;
    int block_size_long = (longMat.rows + thread_size_long - 1)/4;
    // printf("longMat.rows %d long size: %d block_size_long:%d \n",longMat.rows, thread_size_long, block_size_long);
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now(); 

        spmv_fp64<<<block_size_short, thread_size_short, 0, my_stream[0]>>>(short_d_rowPtr, short_d_colIdx, shortMap, shortMat.rows, shortMat.nnz, short_d_values, d_x, d_y);
        spmv_fp64_long<<<block_size_long, thread_size_long, 0, my_stream[1]>>>(long_d_rowPtr, long_d_colIdx, longMap, longMat.rows, longMat.nnz, long_d_values, d_x, d_y);
        cudaStreamSynchronize(my_stream[0]);
        cudaStreamSynchronize(my_stream[1]);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now(); 

        auto elapsed = end - start;
        std::cout << "MySPMV 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    }
    cudaMemcpy(h_y, d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; i++) {
        outvalues[i] = float(h_y[i]);
    }
}