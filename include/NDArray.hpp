#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <vector>
#include <stdexcept>
#include <initializer_list>
#include <iostream>
#include <cmath>

namespace NumCPP {

template<typename T = double>
class NDArray {
public:
    // Constructors and Destructor
    NDArray();  // Default constructor
    ~NDArray(); // Destructor
    NDArray(const NDArray<T>& other); // Copy constructor
    NDArray(NDArray<T>&& other) noexcept; // Move constructor
    NDArray<T>& operator=(const NDArray<T>& other); // Copy assignment
    NDArray<T>& operator=(NDArray<T>&& other) noexcept; // Move assignment
    NDArray(const std::vector<size_t>& shape, const T& init_val = T()); // Constructor with shape
    NDArray(std::initializer_list<size_t> shape, const T& init_val = T()); // Constructor with initializer list

    // Basic Array Properties
    std::vector<size_t> shape() const;
    size_t ndim() const;
    size_t size() const;
    NDArray<T> reshape(const std::vector<size_t>& new_shape) const;
    std::vector<T> flatten() const;
    T sum() const;
    T mean() const;
    T min() const;
    T max() const;
    T determinant() const;
    bool is_square() const;

    // Modify Array
    void fill(const T& value);
    void zeros();
    void ones();
    void transpose();
    void reverse();
    void pow(const T& exponent);
    void invert(); // In-place inversion

    // Return modified array
    NDArray<T> filled(const T& value) const;
    NDArray<T> zeros_like() const;
    NDArray<T> ones_like() const;
    NDArray<T> transposed() const;
    NDArray<T> powed(const T& exponent) const;
    NDArray<T> reversed() const;
    NDArray<T> inverted() const; // Non-modifying inverse
    NDArray<T> kernel() const;
    
    // Return a copy of the array
    NDArray<T> copy() const;

    // Element Access
    T& operator()(int);
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    // Arithmetic Operators (element-wise)
    NDArray<T> operator+(const NDArray<T>& other) const;
    NDArray<T> operator-(const NDArray<T>& other) const;
    NDArray<T> operator*(const NDArray<T>& other) const;
    NDArray<T> operator/(const NDArray<T>& other) const;
    NDArray<T> dot(const NDArray<T>& other) const;

    // Utility
    void print() const;

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    T* data_;  // Pointer to data

    // Helper Functions
    std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const;
    size_t compute_index(const std::vector<size_t>& indices) const;

    // Helper for matrix inversion
    T determinant_helper() const;
    void invert_helper(); // Helper function for in-place inversion
};

} // namespace NumCPP

#include "NDArray.tpp"

#endif // NDARRAY_HPP
