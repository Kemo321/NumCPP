#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Array.hpp"
#include <stdexcept>
#include <vector>

namespace NumCPP {

template <typename T>
class Matrix {
public:
    // Constructor and Destructor
    Matrix();
    ~Matrix();
    Matrix(Array<T>& arr);

    Matrix(const Matrix<T>& other);
    Matrix(Matrix<T>&& other) noexcept;

    Matrix<T>& operator=(const Matrix<T>& other);
    Matrix<T>& operator=(Matrix<T>&& other) noexcept;

    Matrix(const std::vector<size_t>& shape, const T& init_val = T());
    Matrix(std::initializer_list<size_t> shape, const T& init_val = T());

    Matrix(const std::vector<size_t>& shape, const std::vector<T>& data);
    Matrix(std::initializer_list<size_t> shape, const std::vector<T>& data);

    // Basic Matrix Properties
    std::vector<size_t> shape() const;
    size_t ndim() const;
    size_t size() const;
    std::vector<size_t> strides() const;

    // Basic Matrix Operations
    T sum() const;
    T mean() const;
    T min() const;
    T max() const;
    bool is_square() const;
    Matrix<T> reshape(const std::vector<size_t>& new_shape) const;
    Array<T> flatten() const;

    // Modification Methods
    void fill(const T& value);
    void zeros();
    void ones();
    void transpose();
    void reverse();
    void pow(const T& exponent);

    // Return modified matrix
    Matrix<T> filled(const T& value) const;
    Matrix<T> zeros_like() const;
    Matrix<T> ones_like() const;
    Matrix<T> transposed() const;
    Matrix<T> powed(const T& exponent) const;
    Matrix<T> reversed() const;

    // Return a copy of the matrix
    Matrix<T> copy() const;

    // Element Access
    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;
    template <typename... Indices>
    T& operator()(Indices... indices);
    template <typename... Indices>
    const T& operator()(Indices... indices) const;

    // Matrix-specific operations
    Matrix<T> transpose() const;
    Matrix<T> dot(const Matrix<T>& other) const;

    // Arithmetic Operators (element-wise)
    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> operator*(const Matrix<T>& other) const;
    Matrix<T> operator/(const Matrix<T>& other) const;
    Matrix<T> operator+(const T& scalar) const;
    Matrix<T> operator-(const T& scalar) const;
    Matrix<T> operator*(const T& scalar) const;
    Matrix<T> operator/(const T& scalar) const;
    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T>& operator*=(const Matrix<T>& other);
    Matrix<T>& operator/=(const Matrix<T>& other);
    Matrix<T>& operator+=(const T& scalar);
    Matrix<T>& operator-=(const T& scalar);
    Matrix<T>& operator*=(const T& scalar);
    Matrix<T>& operator/=(const T& scalar);
    Matrix<T> operator-() const;
    Matrix<T> operator+() const;
    Matrix<T> operator++();
    Matrix<T> operator--();
    Matrix<T> operator++(int);
    Matrix<T> operator--(int);
    Matrix<T> operator!() const;
    Matrix<T> operator~() const;
    Matrix<T> operator&(const Matrix<T>& other) const;
    Matrix<T> operator|(const Matrix<T>& other) const;
    Matrix<T> operator^(const Matrix<T>& other) const;
    Matrix<T>& operator&=(const Matrix<T>& other);
    Matrix<T>& operator|=(const Matrix<T>& other);
    Matrix<T>& operator^=(const Matrix<T>& other);
    Matrix<T> operator&(const T& scalar) const;
    Matrix<T> operator|(const T& scalar) const;
    Matrix<T> operator^(const T& scalar) const;
    Matrix<T>& operator&=(const T& scalar);
    Matrix<T>& operator|=(const T& scalar);
    Matrix<T>& operator^=(const T& scalar);
    Matrix<T> operator==(const Matrix<T>& other) const;
    Matrix<T> operator!=(const Matrix<T>& other) const;
    Matrix<T> operator<(const Matrix<T>& other) const;
    Matrix<T> operator<=(const Matrix<T>& other) const;
    Matrix<T> operator>(const Matrix<T>& other) const;
    Matrix<T> operator>=(const Matrix<T>& other) const;
    Matrix<T> operator==(const T& scalar) const;
    Matrix<T> operator!=(const T& scalar) const;
    Matrix<T> operator<(const T& scalar) const;
    Matrix<T> operator<=(const T& scalar) const;
    Matrix<T> operator>(const T& scalar) const;
    Matrix<T> operator>=(const T& scalar) const;
    Matrix<T> operator&&(const Matrix<T>& other) const;
    Matrix<T> operator||(const Matrix<T>& other) const;
    Matrix<T> operator&&(const T& scalar) const;
    Matrix<T> operator||(const T& scalar) const;

    // Utility
    void print() const;
    void print_shape() const;
    void print_strides() const;

    // Matrix-specific operation
    Array<T> dot(const Matrix<T>& other) const;

private:
    Array<T>& arr_; // Reference to underlying Array
};

} // namespace NumCPP

#endif // MATRIX_HPP