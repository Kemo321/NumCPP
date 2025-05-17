#ifndef SQUAREMATRIX_TPP
#define SQUAREMATRIX_TPP

#include "SquareMatrix.hpp"
#include <stdexcept>
#include <algorithm>

namespace NumCPP {

// Constructor with size n (creates an n x n matrix)
template <typename T>
SquareMatrix<T>::SquareMatrix(size_t n) : Matrix<T>(n, n), size(n) {}

// Constructor with Array (checks if square)
template <typename T>
SquareMatrix<T>::SquareMatrix(const Array<T>& arr) : Matrix<T>(arr), size(arr.shape()[0]) {
    if (arr.shape()[0] != arr.shape()[1]) {
        throw std::invalid_argument("Array must represent a square matrix");
    }
}

// Determinant using Gaussian elimination
template <typename T>
T SquareMatrix<T>::determinant() const {
    if (size == 0) return T(1);
    Array<T> temp = this->arr_.copy();
    T det = T(1);
    int swaps = 0;

    for (size_t i = 0; i < size; ++i) {
        size_t max_row = i;
        for (size_t k = i + 1; k < size; ++k) {
            if (std::abs(temp({k, i})) > std::abs(temp({max_row, i}))) {
                max_row = k;
            }
        }
        if (std::abs(temp({max_row, i})) < 1e-10) {
            return T(0); // Singular matrix
        }
        if (max_row != i) {
            for (size_t j = 0; j < size; ++j) {
                std::swap(temp({i, j}), temp({max_row, j}));
            }
            swaps++;
        }
        for (size_t k = i + 1; k < size; ++k) {
            T factor = temp({k, i}) / temp({i, i});
            for (size_t j = i; j < size; ++j) {
                temp({k, j}) -= factor * temp({i, j});
            }
        }
        det *= temp({i, i});
    }
    if (swaps % 2 == 1) det = -det;
    return det;
}

// Inverse using Gaussian elimination with augmentation
template <typename T>
SquareMatrix<T> SquareMatrix<T>::inverse() const {
    if (size == 0) throw std::runtime_error("Cannot invert empty matrix");
    Array<T> augmented({size, 2 * size});
    // Initialize augmented matrix [A | I]
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            augmented({i, j}) = this->arr_({i, j});
            augmented({i, j + size}) = (i == j) ? T(1) : T(0);
        }
    }
    // Gaussian elimination with partial pivoting
    for (size_t i = 0; i < size; ++i) {
        size_t max_row = i;
        for (size_t k = i + 1; k < size; ++k) {
            if (std::abs(augmented({k, i})) > std::abs(augmented({max_row, i}))) {
                max_row = k;
            }
        }
        if (std::abs(augmented({max_row, i})) < 1e-10) {
            throw std::runtime_error("Matrix is singular and cannot be inverted");
        }
        if (max_row != i) {
            for (size_t j = 0; j < 2 * size; ++j) {
                std::swap(augmented({i, j}), augmented({max_row, j}));
            }
        }
        T pivot = augmented({i, i});
        for (size_t j = 0; j < 2 * size; ++j) {
            augmented({i, j}) /= pivot;
        }
        for (size_t k = 0; k < size; ++k) {
            if (k != i) {
                T factor = augmented({k, i});
                for (size_t j = 0; j < 2 * size; ++j) {
                    augmented({k, j}) -= factor * augmented({i, j});
                }
            }
        }
    }
    // Extract inverse
    Array<T> inv_arr({size, size});
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            inv_arr({i, j}) = augmented({i, j + size});
        }
    }
    return SquareMatrix<T>(inv_arr);
}

} // namespace NumCPP

#endif // SQUAREMATRIX_TPP