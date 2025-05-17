#ifndef SQUAREMATRIX_HPP
#define SQUAREMATRIX_HPP

#include "Matrix.hpp"
#include <vector>
#include <cmath>

namespace NumCPP {

template <typename T>
class SquareMatrix : public Matrix<T> {
private:
    size_t size; // Store the size for convenience

public:
    // Constructors
    SquareMatrix(size_t n);
    SquareMatrix(const Array<T>& arr);

    // Determinant
    T determinant() const;

    // Inverse
    SquareMatrix<T> inverse() const;
};

} // namespace NumCPP

#include "SquareMatrix.tpp"

#endif // SQUAREMATRIX_HPP