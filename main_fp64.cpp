#include "amgT.h"
#include "ref.h"
#include "cusp.h"
#include "fast_matrix_market/fast_matrix_market.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "matrixFormat.hpp"
using namespace std;
template <typename IT, typename VT>
struct triplet_matrix {
    int64_t nrows = 0, ncols = 0;
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
    options.chunk_size_bytes = 2048;

    fast_matrix_market::read_matrix_market_triplet(f, triplet.nrows, triplet.ncols, triplet.rows, triplet.cols, triplet.vals, options);
}

int main() {
    triplet_matrix<int64_t, double> triplet;
    read_triplet_file("/staff/chengjiajun/workspace/benchsp/data/af_shell3/af_shell3.mtx", triplet);
    vector<int>counts(triplet.nrows+1, 0);
    int nnz = triplet.vals.size();
    printf("nnz: %d nrows: %ld  ncols: %ld\n", nnz, triplet.nrows, triplet.ncols);
    for (int i = 0; i < triplet.vals.size(); i++) {
        counts[triplet.rows[i]+1]++;
    }

    // counts
    for (int i = 1; i <= triplet.nrows; i++) {
        counts[i] += counts[i-1];
    }
    vector<double>x(triplet.nrows, 1);
    vector<double>ref_res(triplet.ncols, 0);
    vector<double>cu_res(triplet.ncols, 0);
    auto start = std::chrono::high_resolution_clock::now(); 
    ref_spmv_fp64(counts.data(), triplet.cols.data(), triplet.vals.data(), x.data(), ref_res.data(), triplet.nrows, triplet.ncols);
    auto end = std::chrono::high_resolution_clock::now(); 
    auto elapsed = end - start;
    std::cout << "baseline 耗时: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
    // start = std::chrono::high_resolution_clock::now(); 
    cusparse_spmv_fp64(counts.data(), triplet.cols.data(), triplet.vals.data(), x.data(), cu_res.data(), triplet.nrows, triplet.ncols, nnz);
    // end = std::chrono::high_resolution_clock::now(); 
    // elapsed = end - start;
    // std::cout << "耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms\n";
    if (verify_res(1e-3, 1e-3, ref_res.data(), cu_res.data(), triplet.nrows)) {
        printf("PASS\n");
    } else {
        printf("FAIL\n");
    }
    return 0;
}