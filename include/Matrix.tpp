#ifndef MATRIX_TPP
#define MATRIX_TPP

#include "Matrix.hpp"
#include <iostream>
#include <stdexcept>

namespace NumCPP {

// Constructors and Destructor
template <typename T>
Matrix<T>::Matrix()
    : arr_(*(new Array<T>({ 0, 0 })))
{
    // Default constructor creates an empty 2D array
}

template <typename T>
Matrix<T>::~Matrix()
{
    // Since arr_ is a reference, we don't delete it here
}

template <typename T>
Matrix<T>::Matrix(Array<T>& arr)
    : arr_(arr)
{
    if (arr.ndim() != 2)
        throw std::invalid_argument("Array must be 2D for Matrix");
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& other)
    : arr_(other.arr_)
{
}

template <typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept
    : arr_(other.arr_)
{
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
    if (this != &other) {
        arr_ = other.arr_; // Reference reassignment
    }
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept
{
    if (this != &other) {
        arr_ = other.arr_; // Reference reassignment
    }
    return *this;
}

template <typename T>
Matrix<T>::Matrix(const std::vector<size_t>& shape, const T& init_val)
    : arr_(*(new Array<T>(shape, init_val)))
{
    if (shape.size() != 2)
        throw std::invalid_argument("Matrix must be 2D");
}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<size_t> shape, const T& init_val)
    : arr_(*(new Array<T>(shape, init_val)))
{
    if (shape.size() != 2)
        throw std::invalid_argument("Matrix must be 2D");
}

template <typename T>
Matrix<T>::Matrix(const std::vector<size_t>& shape, const std::vector<T>& data)
    : arr_(*(new Array<T>(shape, data)))
{
    if (shape.size() != 2)
        throw std::invalid_argument("Matrix must be 2D");
}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<size_t> shape, const std::vector<T>& data)
    : arr_(*(new Array<T>(shape, data)))
{
    if (shape.size() != 2)
        throw std::invalid_argument("Matrix must be 2D");
}

// Basic Matrix Properties
template <typename T>
std::vector<size_t> Matrix<T>::shape() const
{
    return arr_.shape();
}

template <typename T>
size_t Matrix<T>::ndim() const
{
    return arr_.ndim(); // Always 2 for Matrix
}

template <typename T>
size_t Matrix<T>::size() const
{
    return arr_.size();
}

template <typename T>
std::vector<size_t> Matrix<T>::strides() const
{
    return arr_.strides();
}

// Basic Matrix Operations
template <typename T>
T Matrix<T>::sum() const
{
    return arr_.sum();
}

template <typename T>
T Matrix<T>::mean() const
{
    return arr_.mean();
}

template <typename T>
T Matrix<T>::min() const
{
    return arr_.min();
}

template <typename T>
T Matrix<T>::max() const
{
    return arr_.max();
}

template <typename T>
bool Matrix<T>::is_square() const
{
    return arr_.is_square();
}

template <typename T>
Matrix<T> Matrix<T>::reshape(const std::vector<size_t>& new_shape) const
{
    if (new_shape.size() != 2)
        throw std::invalid_argument("Matrix must be 2D");
    Array<T>& new_arr = *(new Array<T>(arr_.reshape(new_shape)));
    return Matrix<T>(new_arr);
}

template <typename T>
Array<T> Matrix<T>::flatten() const
{
    return arr_.flatten();
}

// Modification Methods
template <typename T>
void Matrix<T>::fill(const T& value)
{
    arr_.fill(value);
}

template <typename T>
void Matrix<T>::zeros()
{
    arr_.zeros();
}

template <typename T>
void Matrix<T>::ones()
{
    arr_.ones();
}

template <typename T>
void Matrix<T>::transpose()
{
    arr_.transpose();
}

template <typename T>
void Matrix<T>::reverse()
{
    arr_.reverse();
}

template <typename T>
void Matrix<T>::pow(const T& exponent)
{
    arr_.pow(exponent);
}

// Return Modified Matrix
template <typename T>
Matrix<T> Matrix<T>::filled(const T& value) const
{
    Array<T>& new_arr = *(new Array<T>(arr_.filled(value)));
    return Matrix<T>(new_arr);
}

template <typename T>
Matrix<T> Matrix<T>::zeros_like() const
{
    Array<T>& new_arr = *(new Array<T>(arr_.zeros_like()));
    return Matrix<T>(new_arr);
}

template <typename T>
Matrix<T> Matrix<T>::ones_like() const
{
    Array<T>& new_arr = *(new Array<T>(arr_.ones_like()));
    return Matrix<T>(new_arr);
}

template <typename T>
Matrix<T> Matrix<T>::transposed() const
{
    Array<T>& new_arr = *(new Array<T>(arr_.transposed()));
    return Matrix<T>(new_arr);
}

template <typename T>
Matrix<T> Matrix<T>::powed(const T& exponent) const
{
    Array<T>& new_arr = *(new Array<T>(arr_.powed(exponent)));
    return Matrix<T>(new_arr);
}

template <typename T>
Matrix<T> Matrix<T>::reversed() const
{
    Array<T>& new_arr = *(new Array<T>(arr_.reversed()));
    return Matrix<T>(new_arr);
}

// Return a Copy of the Matrix
template <typename T>
Matrix<T> Matrix<T>::copy() const
{
    Array<T>& new_arr = *(new Array<T>(arr_.copy()));
    return Matrix<T>(new_arr);
}

// Element Access
template <typename T>
T& Matrix<T>::operator()(size_t row, size_t col)
{
    return arr_({ row, col });
}

template <typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const
{
    return arr_({ row, col });
}

template <typename T>
T& Matrix<T>::operator[](size_t index)
{
    return arr_[index];
}

template <typename T>
const T& Matrix<T>::operator[](size_t index) const
{
    return arr_[index];
}

template <typename T>
T& Matrix<T>::operator()(const std::vector<size_t>& indices)
{
    if (indices.size() != 2)
        throw std::invalid_argument("Matrix requires 2 indices");
    return arr_(indices);
}

template <typename T>
const T& Matrix<T>::operator()(const std::vector<size_t>& indices) const
{
    if (indices.size() != 2)
        throw std::invalid_argument("Matrix requires 2 indices");
    return arr_(indices);
}

template <typename T>
template <typename... Indices>
T& Matrix<T>::operator()(Indices... indices)
{
    static_assert(sizeof...(indices) == 2, "Matrix requires exactly 2 indices");
    return arr_(indices...);
}

template <typename T>
template <typename... Indices>
const T& Matrix<T>::operator()(Indices... indices) const
{
    static_assert(sizeof...(indices) == 2, "Matrix requires exactly 2 indices");
    return arr_(indices...);
}

// Matrix-Specific Operations
template <typename T>
Matrix<T> Matrix<T>::transpose() const
{
    return transposed(); // Delegates to transposed() for consistency
}

template <typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T>& other) const
{
    const auto& shape1 = arr_.shape();
    const auto& shape2 = other.arr_.shape();
    if (shape1[1] != shape2[0])
        throw std::runtime_error("Shapes do not align for dot product");
    size_t m = shape1[0];
    size_t n = shape1[1];
    size_t p = shape2[1];
    Array<T> result({ m, p }, T(0));

    size_t total_elements = m * p;
    if (total_elements > 1000) {
// Multithreaded implementation
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                for (size_t k = 0; k < n; ++k) {
#pragma omp atomic
                    result({ i, j }) += arr_({ i, k }) * other.arr_({ k, j });
                }
            }
        }
    } else {
        // Single-threaded implementation
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    result({ i, j }) += arr_({ i, k }) * other.arr_({ k, j });
                }
            }
        }
    }
    return result;
}

// Arithmetic Operators (Element-wise)
template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for addition");
    Array<T>& result = *(new Array<T>(arr_ + other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for subtraction");
    Array<T>& result = *(new Array<T>(arr_ - other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for multiplication");
    Array<T>& result = *(new Array<T>(arr_ * other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for division");
    Array<T>& result = *(new Array<T>(arr_ / other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ + scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ - scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ * scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ / scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other)
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for addition");
    arr_ += other.arr_;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other)
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for subtraction");
    arr_ -= other.arr_;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& other)
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for multiplication");
    arr_ *= other.arr_;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/=(const Matrix<T>& other)
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for division");
    arr_ /= other.arr_;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const T& scalar)
{
    arr_ += scalar;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const T& scalar)
{
    arr_ -= scalar;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar)
{
    arr_ *= scalar;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/=(const T& scalar)
{
    arr_ /= scalar;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-() const
{
    Array<T>& result = *(new Array<T>(-arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator+() const
{
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator++()
{
    arr_++;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator--()
{
    arr_--;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator++(int)
{
    Matrix<T> temp(*this);
    arr_++;
    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::operator--(int)
{
    Matrix<T> temp(*this);
    arr_--;
    return temp;
}

template <typename T>
Matrix<T> Matrix<T>::operator!() const
{
    Array<T>& result = *(new Array<T>(!arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator~() const
{
    Array<T>& result = *(new Array<T>(~arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator&(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for bitwise AND");
    Array<T>& result = *(new Array<T>(arr_ & other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator|(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for bitwise OR");
    Array<T>& result = *(new Array<T>(arr_ | other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator^(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for bitwise XOR");
    Array<T>& result = *(new Array<T>(arr_ ^ other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T>& Matrix<T>::operator&=(const Matrix<T>& other)
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for bitwise AND");
    arr_ &= other.arr_;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator|=(const Matrix<T>& other)
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for bitwise OR");
    arr_ |= other.arr_;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator^=(const Matrix<T>& other)
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for bitwise XOR");
    arr_ ^= other.arr_;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator&(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ & scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator|(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ | scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator^(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ ^ scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T>& Matrix<T>::operator&=(const T& scalar)
{
    arr_ &= scalar;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator|=(const T& scalar)
{
    arr_ |= scalar;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator^=(const T& scalar)
{
    arr_ ^= scalar;
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator==(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for equality comparison");
    Array<T>& result = *(new Array<T>(arr_ == other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator!=(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for inequality comparison");
    Array<T>& result = *(new Array<T>(arr_ != other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator<(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for less-than comparison");
    Array<T>& result = *(new Array<T>(arr_ < other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator<=(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for less-than-or-equal comparison");
    Array<T>& result = *(new Array<T>(arr_ <= other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator>(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for greater-than comparison");
    Array<T>& result = *(new Array<T>(arr_ > other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator>=(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for greater-than-or-equal comparison");
    Array<T>& result = *(new Array<T>(arr_ >= other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator==(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ == scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator!=(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ != scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator<(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ < scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator<=(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ <= scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator>(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ > scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator>=(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ >= scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator&&(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for logical AND");
    Array<T>& result = *(new Array<T>(arr_ && other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator||(const Matrix<T>& other) const
{
    if (shape() != other.shape())
        throw std::runtime_error("Shapes do not match for logical OR");
    Array<T>& result = *(new Array<T>(arr_ || other.arr_));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator&&(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ && scalar));
    return Matrix<T>(result);
}

template <typename T>
Matrix<T> Matrix<T>::operator||(const T& scalar) const
{
    Array<T>& result = *(new Array<T>(arr_ || scalar));
    return Matrix<T>(result);
}

// Utility
template <typename T>
void Matrix<T>::print() const
{
    arr_.print();
}

template <typename T>
void Matrix<T>::print_shape() const
{
    const auto& s = shape();
    std::cout << "Shape: [" << s[0] << ", " << s[1] << "]" << std::endl;
}

template <typename T>
void Matrix<T>::print_strides() const
{
    const auto& st = strides();
    std::cout << "Strides: [" << st[0] << ", " << st[1] << "]" << std::endl;
}

} // namespace NumCPP

#endif // MATRIX_TPP