#include "amgT.h"
#include "ref.h"
#include "bcsr.h"
#include "fast_matrix_market/fast_matrix_market.hpp"

#include <iostream>
#include "cusp.h"
#include "blas.h"
#include <fstream>
#include "spmv.h"
#include <vector>
#include <chrono>
#include "dasp.h"
#include <algorithm>
using namespace std;
template <typename IT, typename VT>
struct triplet_matrix {
    int32_t nrows = 0, ncols = 0;
    std::vector<IT> rows;
    std::vector<IT> cols;
    std::vector<VT> vals;
};

bool verify_res(double rtol, double atol, double *res, double *ref_res, int nnz) {
    for (int i = 0; i < nnz; i++) {
        double diff = fabsf(res[i] - ref_res[i]);
        double tol = atol + rtol * fabsf(ref_res[i]);
        if (diff > tol) {
            printf("Error: %d %f %f %f\n", i, res[i], ref_res[i], diff);
            return false; // 任何元素超出容差即返回 false
        }
    }
    return true;
}

template <typename TRIPLET>
void read_triplet_file(const std::string& matrix_filename, TRIPLET& triplet, fast_matrix_market::read_options options = {}) {
    std::ifstream f( matrix_filename);
    options.chunk_size_bytes = 1024*1024;

    fast_matrix_market::read_matrix_market_triplet(f, triplet.nrows, triplet.ncols, triplet.rows, triplet.cols, triplet.vals, options);
}

int main(int argc, char **argv) {
    triplet_matrix<int32_t, double> triplet;
        std::string file_path = argv[1];
    // read_triplet_file("/staff/chengjiajun/workspace/benchsp/data/mycielskian3/mycielskian3.mtx", triplet);
    read_triplet_file(file_path, triplet);
    int repeat = atoi(argv[2]);
    // ifstream ifs("/staff/chengjiajun/workspace/benchsp/data/mycielskian3/mycielskian3.mtx");
    // fast_matrix_market::read_matrix_market_triplet(ifs, triplet.nrows, triplet.ncols, triplet.rows, triplet.cols, triplet.vals);
    int32_t* counts = (int32_t *)malloc((triplet.nrows+1) * sizeof(int32_t));
    memset(counts, 0, (triplet.nrows+1) * sizeof(int32_t));
    // vector<int>counts(triplet.nrows+1, 0);

    // Eigen::SparseMatrix<double> mat;
    // fast_matrix_market::read_matrix_market_eigen(ifs, mat);
    int32_t nnz = triplet.vals.size();
    // printf("nnz: %d nrows: %ld  ncols: %ld\n", nnz, triplet.nrows, triplet.ncols);
    vector<vector<int32_t>> datas;
    for (int i = 0; i < nnz; i++) {
        datas.push_back(vector<int32_t>{triplet.rows[i], triplet.cols[i], triplet.vals[i]});
    }
    sort(datas.begin(), datas.end(), [](const vector<int32_t>& a, const vector<int32_t>& b) {
        if (a[0] == b[0]) {
            return a[1] < b[1];
        }
        return a[0] < b[0];
    });
    for (int i = 0; i < nnz; i++) {
        triplet.rows[i] = datas[i][0];
        triplet.cols[i] = datas[i][1];
        triplet.vals[i] = datas[i][2];
    }
    for (int i = 0; i < triplet.vals.size(); i++) {
        counts[triplet.rows[i]+1]++;
    }
    // for (int i = 0; i < 2; i++) {
    //     printf("counts[%d]: %d\n", i, counts[i]);
    // }
    // counts
    for (int i = 1; i <= triplet.nrows; i++) {
        counts[i] += counts[i-1];
    }
    double *x = (double *)malloc(triplet.nrows * sizeof(double));
    // vector<double>x(triplet.nrows, 1);
    for (int i = 0;i  < triplet.nrows; i++) {
        if (i%4) {
            x[i] = 1.0;
        } else {
            x[i] = 1.0;
        }
    }

    // vector<double>ref_res(triplet.ncols, 0);
    // vector<double>cu_res(triplet.ncols, 0);
    double *ref_res = (double *)malloc(triplet.ncols * sizeof(double));
    double *cu_res = (double *)malloc(triplet.ncols * sizeof(double));
    memset(ref_res, 0, triplet.ncols * sizeof(double));
    memset(cu_res, 0, triplet.ncols * sizeof(double));
    int32_t *cols = (int32_t *)malloc(nnz * sizeof(int32_t));
    memcpy(cols, triplet.cols.data(), nnz * sizeof(int32_t));
    double *vals = (double *)malloc(nnz * sizeof(double));
    memcpy(vals, triplet.vals.data(), nnz * sizeof(double));

    int64_t* count64 = (int64_t *)malloc((triplet.nrows+1) * sizeof(int64_t));
    int64_t* cols64 = (int64_t *)malloc(nnz * sizeof(int64_t));
    for (int i = 0; i < nnz; i++) cols64[i] = triplet.cols[i];
    for (int i = 0; i <= triplet.nrows; i++) count64[i] = counts[i];
    // ref_spmv_fp64(count64, cols64, vals, x, ref_res, triplet.nrows, triplet.ncols);
    // double *ref_cures;
    // cudaMalloc((void**)&ref_cures, triplet.nrows * sizeof(double));
    // cudaMemset(ref_cures, 0.0,  triplet.nrows * sizeof(double));
    for (int i = 0; i < 1; i++)
    cusparse_spmv_fp64(count64, cols64, vals, x, ref_res, triplet.nrows, triplet.ncols, nnz, repeat);
    // cudaMemcpy(ref_res, ref_cures, triplet.nrows * sizeof(double), cudaMemcpyDeviceToHost);
    // auto end = std::chrono::high_resolution_clock::now(); 
    // auto elapsed = end - start;
    // std::cout << "耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms\n";
    // bcsr_spmv_fp64(counts, cols, vals, x,cu_res, triplet.nrows, triplet.ncols, nnz, repeat);
    // printf("amgT spmv\n");
    int *new_order = (int *)malloc(sizeof(int) * triplet.nrows);
    for (int i = 0; i < 1; i++)
    amgT_spmv_fp64(counts, cols, vals, x, cu_res, triplet.nrows, triplet.ncols, nnz, repeat);

    spmv_all("filename", vals, counts, cols, x, cu_res, new_order, triplet.nrows, triplet.ncols, nnz, 4, 0.75, 256);

    // double *dense_a;
    // dense_a = (double *)malloc(triplet.nrows * triplet.ncols * sizeof(double));
    // memset(dense_a, 0, triplet.nrows * triplet.ncols * sizeof(double));
    // for (int i = 0; i < nnz; i++) {
    //     dense_a[triplet.rows[i] * triplet.ncols + triplet.cols[i]] = triplet.vals[i];
    // }
    // gemvfp64(dense_a, x, cu_res, triplet.nrows, triplet.ncols, 1, repeat);
    // if (verify_res(1e-3, 1e-3, cu_res,ref_res, triplet.nrows)) {
    //     printf("PASS\n");
    // } else {
    //     printf("FAIL\n");
    // }
    // return 0;
}