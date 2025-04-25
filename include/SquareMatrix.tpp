#ifndef SQUAREMATRIX_TPP
#define SQUAREMATRIX_TPP

#include "SquareMatrix.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace NumCPP {

template<typename T>
SquareMatrix<T>::SquareMatrix(Array<T>& arr) : arr_(arr) {
    if (arr.ndim() != 2 || !arr.is_square()) {
        throw std::invalid_argument("Array must be a square 2D array for SquareMatrix operations.");
    }
}

template<typename T>
T SquareMatrix<T>::determinant() const {
    size_t n = arr_.shape()[0];
    if (n == 1) return arr_({0, 0});
    if (n == 2) return arr_({0, 0}) * arr_({1, 1}) - arr_({0, 1}) * arr_({1, 0});
    return determinant_helper();
}

template<typename T>
void SquareMatrix<T>::invert() {
    size_t n = arr_.shape()[0];
    if (n == 1) {
        arr_({0, 0}) = 1 / arr_({0, 0});
        return;
    }
    if (n == 2) {
        T det = arr_({0, 0}) * arr_({1, 1}) - arr_({0, 1}) * arr_({1, 0});
        if (std::abs(det) < 1e-10) throw std::runtime_error("Matrix is singular and cannot be inverted.");
        T temp = arr_({0, 0});
        arr_({0, 0}) = arr_({1, 1}) / det;
        arr_({0, 1}) = -arr_({0, 1}) / det;
        arr_({1, 0}) = -arr_({1, 0}) / det;
        arr_({1, 1}) = temp / det;
        return;
    }
    invert_helper();
}

template<typename T>
Array<T> SquareMatrix<T>::inverted() const {
    Array<T> copy = arr_.copy();
    SquareMatrix<T> sq_copy(copy);
    sq_copy.invert();
    return copy;
}

template<typename T>
T SquareMatrix<T>::determinant_helper() const {
    size_t n = arr_.shape()[0];
    Array<T> mat = arr_.copy();
    T det = 1;
    for (size_t i = 0; i < n; ++i) {
        size_t pivotRow = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(mat({j, i})) > std::abs(mat({pivotRow, i}))) pivotRow = j;
        }
        if (std::abs(mat({pivotRow, i})) < 1e-10) return 0;
        if (i != pivotRow) {
            for (size_t k = 0; k < n; ++k) {
                std::swap(mat({pivotRow, k}), mat({i, k}));
            }
            det = -det;
        }
        for (size_t j = i + 1; j < n; ++j) {
            T factor = mat({j, i}) / mat({i, i});
            for (size_t k = i; k < n; ++k) {
                mat({j, k}) -= factor * mat({i, k});
            }
        }
        det *= mat({i, i});
    }
    return det;
}

template<typename T>
void SquareMatrix<T>::invert_helper() {
    size_t n = arr_.shape()[0];
    Array<T> mat = arr_.copy();
    Array<T> inv({n, n}, T(0));
    for (size_t i = 0; i < n; ++i) inv({i, i}) = 1;
    for (size_t i = 0; i < n; ++i) {
        size_t pivotRow = i;
        T maxPivot = std::abs(mat({i, i}));
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(mat({j, i})) > maxPivot) {
                maxPivot = std::abs(mat({j, i}));
                pivotRow = j;
            }
        }
        if (maxPivot < 1e-10) throw std::runtime_error("Matrix is singular and cannot be inverted.");
        if (i != pivotRow) {
            for (size_t k = 0; k < n; ++k) {
                std::swap(mat({pivotRow, k}), mat({i, k}));
                std::swap(inv({pivotRow, k}), inv({i, k}));
            }
        }
        T pivot = mat({i, i});
        for (size_t k = 0; k < n; ++k) {
            mat({i, k}) /= pivot;
            inv({i, k}) /= pivot;
        }
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                T factor = mat({j, i});
                for (size_t k = 0; k < n; ++k) {
                    mat({j, k}) -= factor * mat({i, k});
                    inv({j, k}) -= factor * inv({i, k});
                }
            }
        }
    }
    arr_ = inv;
}

} // namespace NumCPP

#endif // SQUAREMATRIX_TPP