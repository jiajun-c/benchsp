#include "amgT.h"
#include "ref.h"
#include "cusp.h"
#include "fast_matrix_market/fast_matrix_market.hpp"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;
template <typename IT, typename VT>
struct triplet_matrix {
    int64_t nrows = 0, ncols = 0;
    std::vector<IT> rows;
    std::vector<IT> cols;
    std::vector<VT> vals;
};


template <typename TRIPLET>
void read_triplet_file(const std::string& matrix_filename, TRIPLET& triplet, fast_matrix_market::read_options options = {}) {
    std::ifstream f( matrix_filename);
    options.chunk_size_bytes = 2048;

    fast_matrix_market::read_matrix_market_triplet(f, triplet.nrows, triplet.ncols, triplet.rows, triplet.cols, triplet.vals, options);
}

int main() {
    triplet_matrix<int64_t, float> triplet;
    read_triplet_file("/staff/chengjiajun/workspace/benchsp/data/bcspwr01/bcspwr01.mtx", triplet);
    vector<int64_t>counts(triplet.nrows+1);
    int nnz = triplet.vals.size();
    printf("nnz = %d nrows: %ld  ncols: %ld\n", nnz, triplet.nrows, triplet.ncols);
    for (int i = 0; i < triplet.vals.size(); i++) {
        counts[triplet.rows[i]]++;
    }
    for (int i = 1; i <= triplet.nrows; i++) {
        counts[i] += counts[i-1];
    }
    vector<float>x(triplet.nrows, 1);
    vector<float>ref_res(triplet.ncols, 0);

    ref_spmv(counts.data(), triplet.cols.data(), triplet.vals.data(), x.data(), ref_res.data(), triplet.nrows, triplet.ncols);
    int sum = 0;
    for (int i =0; i < triplet.nrows; i++) {
        sum += ref_res[i];
    }
    printf("sum %d\n", sum);
    return 0;
}