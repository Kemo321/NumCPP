#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace NumCPP {

template <typename T = double>
class Array {
public:
    // Constructors and Destructor
    Array();
    ~Array();
    Array(const Array<T>& other);
    Array(Array<T>&& other) noexcept;
    Array<T>& operator=(const Array<T>& other);
    Array<T>& operator=(Array<T>&& other) noexcept;
    Array(const std::vector<size_t>& shape, const T& init_val = T());
    Array(std::initializer_list<size_t> shape, const T& init_val = T());
    Array(const std::vector<size_t>& shape, const std::vector<T>& data);
    Array(std::initializer_list<size_t> shape, const std::vector<T>& data);

    // Basic Array Properties
    std::vector<size_t> shape() const;
    size_t ndim() const;
    size_t size() const;
    std::vector<size_t> strides() const;
    Array<T> reshape(const std::vector<size_t>& new_shape) const;
    std::vector<T> flatten() const;
    T sum() const;
    T mean() const;
    T min() const;
    T max() const;
    bool is_square() const;

    // Modify Array
    void fill(const T& value);
    void zeros();
    void ones();
    void transpose();
    void reverse();
    void pow(const T& exponent);

    // Return modified array
    Array<T> filled(const T& value) const;
    Array<T> zeros_like() const;
    Array<T> ones_like() const;
    Array<T> transposed() const;
    Array<T> powed(const T& exponent) const;
    Array<T> reversed() const;

    // Return a copy of the array
    Array<T> copy() const;

    // Element Access
    T& operator()(size_t index);
    const T& operator()(size_t index) const;
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;
    template <typename... Indices>
    T& operator()(Indices... indices);
    template <typename... Indices>
    const T& operator()(Indices... indices) const;

    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    T& operator[](const std::vector<size_t>& indices);
    const T& operator[](const std::vector<size_t>& indices) const;

    // Arithmetic Operators (element-wise)
    Array<T> operator+(const Array<T>& other) const;
    Array<T> operator-(const Array<T>& other) const;
    Array<T> operator*(const Array<T>& other) const;
    Array<T> operator/(const Array<T>& other) const;
    Array<T> operator+(const T& scalar) const;
    Array<T> operator-(const T& scalar) const;
    Array<T> operator*(const T& scalar) const;
    Array<T> operator/(const T& scalar) const;
    Array<T>& operator+=(const Array<T>& other);
    Array<T>& operator-=(const Array<T>& other);
    Array<T>& operator*=(const Array<T>& other);
    Array<T>& operator/=(const Array<T>& other);
    Array<T>& operator+=(const T& scalar);
    Array<T>& operator-=(const T& scalar);
    Array<T>& operator*=(const T& scalar);
    Array<T>& operator/=(const T& scalar);
    Array<T> operator-() const;
    Array<T> operator+() const;
    Array<T> operator++();
    Array<T> operator--();
    Array<T> operator++(int);
    Array<T> operator--(int);
    Array<T> operator!() const;
    Array<T> operator~() const;
    Array<T> operator&() const;
    Array<T>& operator&();
    Array<T> operator&(const Array<T>& other) const;
    Array<T> operator|(const Array<T>& other) const;
    Array<T> operator^(const Array<T>& other) const;
    Array<T>& operator&=(const Array<T>& other);
    Array<T>& operator|=(const Array<T>& other);
    Array<T>& operator^=(const Array<T>& other);
    Array<T> operator&(const T& scalar) const;
    Array<T> operator|(const T& scalar) const;
    Array<T> operator^(const T& scalar) const;
    Array<T>& operator&=(const T& scalar);
    Array<T>& operator|=(const T& scalar);
    Array<T>& operator^=(const T& scalar);
    Array<T> operator==(const Array<T>& other) const;
    Array<T> operator!=(const Array<T>& other) const;
    Array<T> operator<(const Array<T>& other) const;
    Array<T> operator<=(const Array<T>& other) const;
    Array<T> operator>(const Array<T>& other) const;
    Array<T> operator>=(const Array<T>& other) const;
    Array<T> operator==(const T& scalar) const;
    Array<T> operator!=(const T& scalar) const;
    Array<T> operator<(const T& scalar) const;
    Array<T> operator<=(const T& scalar) const;
    Array<T> operator>(const T& scalar) const;
    Array<T> operator>=(const T& scalar) const;
    Array<T> operator&&(const Array<T>& other) const;
    Array<T> operator||(const Array<T>& other) const;
    Array<T> operator&&(const T& scalar) const;
    Array<T> operator||(const T& scalar) const;

    // Utility
    void print() const;

protected:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    T* data_;

    // Helper Functions
    std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const;
    size_t compute_index(const std::vector<size_t>& indices) const;
};

} // namespace NumCPP

#include "Array.tpp"

#endif // ARRAY_HPP