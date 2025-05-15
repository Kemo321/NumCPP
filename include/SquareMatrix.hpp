#ifndef SQUAREMATRIX_HPP
#define SQUAREMATRIX_HPP

#include "Array.hpp"

namespace NumCPP {

template <typename T>
class SquareMatrix {
public:
    SquareMatrix(Array<T>& arr);
    T determinant() const;
    void invert();
    Array<T> inverted() const;

private:
    Array<T>& arr_;
    T determinant_helper() const;
    void invert_helper();
};

} // namespace NumCPP

#include "SquareMatrix.tpp"

#endif // SQUAREMATRIX_HPP