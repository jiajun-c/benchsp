#include "amgT.h"
#include "ref.h"
#include "cusp.h"
#include "fast_matrix_market/fast_matrix_market.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include "smat.h"
#include <chrono>
using namespace std;
template <typename IT, typename VT>
struct triplet_matrix {
    int64_t nrows = 0, ncols = 0;
    std::vector<IT> rows;
    std::vector<IT> cols;
    std::vector<VT> vals;
};

bool verify_res(float rtol, float atol, float *res, float *ref_res, int nnz) {
    for (int i = 0; i < nnz; i++) {
        float diff = fabsf(res[i] - ref_res[i]);
        float tol = atol + rtol * fabsf(ref_res[i]);
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

int main() {
    triplet_matrix<int32_t, float> triplet;
    read_triplet_file("/staff/chengjiajun/workspace/benchsp/data/mip1/mip1.mtx", triplet);
    printf("fininsh reading\n");
    // return 0;
    vector<int>counts(triplet.nrows+1, 0);
    int nnz = triplet.vals.size();
    printf("nnz: %d nrows: %ld  ncols: %ld\n", nnz, triplet.nrows, triplet.ncols);
    for (int i = 0; i < triplet.vals.size(); i++) {
        counts[triplet.rows[i]+1]++;
    }
    for (int i = 1; i <= triplet.nrows; i++) {
        counts[i] += counts[i-1];
    }
    for (int n = 256; n <= 2048; n*=2) {
        printf("m: %d n: %d k:%d nnz:%d\n", triplet.nrows, n, triplet.ncols, nnz);
        vector<float>ref_res(triplet.nrows*n, 0);
        vector<float>cu_res(triplet.nrows*n, 0);
        vector<float>x(triplet.ncols*n, 0);
    
        // auto start = std::chrono::high_resolution_clock::now(); 
        for (int i = 0; i < 10; i++)
        // cusparse_spmv_fp16(counts.data(), triplet.cols.data(), triplet.vals.data(), x.data(), cu_res.data(), triplet.nrows, n, triplet.ncols, nnz);
        // auto end = std::chrono::high_resolution_clock::now(); 
        // auto elapsed = end - start;
        // std::cout << "耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms\n";
        // start = std::chrono::high_resolution_clock::now(); 
        printf("\n");
        for (int i = 0; i < 10; i++)
        smatSpmmfp16(counts.data(), triplet.cols.data(), triplet.nrows, triplet.ncols, n, nnz,  triplet.vals.data(), cu_res.data());
       
    }

 // end = std::chrono::high_resolution_clock::now(); 
    // elapsed = end - start;
    // std::cout << "耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms\n";
    // if (verify_res(1e-3, 1e-3, ref_res.data(), cu_res.data(), triplet.nrows)) {
    //     printf("PASS\n");
    // } else {
    //     printf("FAIL\n");
    // }

    return 0;
}