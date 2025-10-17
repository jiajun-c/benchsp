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
#include "csrspmv.h"
#include <map>
using namespace std;
template <typename IT, typename VT>
struct triplet_matrix {
    int32_t nrows = 0, ncols = 0;
    std::vector<IT> rows;
    std::vector<IT> cols;
    std::vector<VT> vals;
};

bool verify_res(double rtol, double atol, double *res, double *ref_res, int nnz) {
    for (int i = 221; i < nnz; i++) {
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
    // ifstream ifs("/staff/chengjiajun/workspace/benchsp/data/mycielskian3/mycielskian3.mtx");
    // fast_matrix_market::read_matrix_market_triplet(ifs, triplet.nrows, triplet.ncols, triplet.rows, triplet.cols, triplet.vals);
    int32_t* counts = (int32_t *)malloc((triplet.nrows+1) * sizeof(int32_t));
    memset(counts, 0, (triplet.nrows+1) * sizeof(int32_t));

    int32_t nnz = triplet.vals.size();
    printf("====matrix info====\n");
    printf("nnz: %d nrows: %ld  ncols: %ld\n", nnz, triplet.nrows, triplet.ncols);
    printf("avglen: %d\n", nnz/triplet.nrows);
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
    int block_len = 4;
    map<int, int>blockinfo;
    for (int i = 0; i < nnz; i++) {
        blockinfo[triplet.rows[i]/block_len * 1000005 + triplet.cols[i]/block_len]++;
    }
    double sum = 0;
    for (auto b: blockinfo) {
        sum += b.second;
    }
    printf("===block info(without recorder)====\n");
    printf("block shape:%d %d\n", block_len, block_len);
    printf("nnz:%d block num:%d avg nnz in block:%f\n",nnz,blockinfo.size(), sum/blockinfo.size());


    map<int, int>blockinforecorder;
    for (int i = 0; i < triplet.nrows; i++) {
        for (int j = counts[i]; j < counts[i+1]; j++) {
            blockinforecorder[triplet.rows[j]/block_len * 1000005 + (j - counts[i])/block_len]++;
        }
    }

    printf("===block info(with recorder)====\n");
    printf("block shape:%d %d\n", block_len, block_len);
    printf("nnz:%d block num:%d avg nnz in block:%f\n",nnz,blockinforecorder.size(), double(nnz)/blockinforecorder.size());

    return 0;
}