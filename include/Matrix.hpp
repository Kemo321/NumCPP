#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Array.hpp"

namespace NumCPP {

template<typename T>
class Matrix {
public:
    Matrix(Array<T>& arr);
    Array<T> dot(const Matrix<T>& other) const;

private:
    Array<T>& arr_;
};

} // namespace NumCPP

#include "Matrix.tpp"

#endif // MATRIX_HPP