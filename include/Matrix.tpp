#ifndef MATRIX_TPP
#define MATRIX_TPP

#include "Matrix.hpp"
#include <stdexcept>
#include <thread>

namespace NumCPP {

template<typename T>
Matrix<T>::Matrix(Array<T>& arr) : arr_(arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("Array must be 2D for Matrix operations.");
    }
}

template<typename T>
Array<T> Matrix<T>::dot(const Matrix<T>& other) const {
    const Array<T>& other_arr = other.arr_;
    if (arr_.shape()[1] != other_arr.shape()[0]) {
        throw std::runtime_error("Shapes do not align for dot product");
    }
    size_t m = arr_.shape()[0];
    size_t n = arr_.shape()[1];
    size_t p = other_arr.shape()[1];
    Array<T> result({m, p}, T(0));
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = m / nthreads;
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < nthreads; t++) {
        size_t row_start = t * block;
        size_t row_end = (t == nthreads - 1) ? m : row_start + block;
        threads.push_back(std::thread([&, this]() {
            for (size_t i = row_start; i < row_end; i++) {
                for (size_t j = 0; j < p; j++) {
                    T sum = T(0);
                    for (size_t k = 0; k < n; k++) {
                        sum += arr_({i, k}) * other_arr({k, j});
                    }
                    result({i, j}) = sum;
                }
            }
        }));
    }
    for (auto &t : threads) t.join();
    return result;
}

} // namespace NumCPP

#endif // MATRIX_TPP