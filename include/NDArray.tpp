#ifndef NDARRAY_TPP
#define NDARRAY_TPP

#include "NDArray.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include <thread>

namespace NumCPP {

//
// Rule of Five
//

template<typename T>
NDArray<T>::NDArray() : data_(nullptr) {
    // Default constructor: creates an empty array.
}

template<typename T>
NDArray<T>::~NDArray() {
    delete[] data_;
}

template<typename T>
NDArray<T>::NDArray(const NDArray<T>& other)
    : shape_(other.shape_), strides_(other.strides_) {
    size_t total = other.size();
    data_ = new T[total];
    std::copy(other.data_, other.data_ + total, data_);
}

template<typename T>
NDArray<T>::NDArray(NDArray<T>&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      data_(other.data_) {
    other.data_ = nullptr;
}

template<typename T>
NDArray<T>& NDArray<T>::operator=(const NDArray<T>& other) {
    if (this != &other) {
        delete[] data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        size_t total = other.size();
        data_ = new T[total];
        std::copy(other.data_, other.data_ + total, data_);
    }
    return *this;
}

template<typename T>
NDArray<T>& NDArray<T>::operator=(NDArray<T>&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        data_ = other.data_;
        other.data_ = nullptr;
    }
    return *this;
}

template<typename T>
NDArray<T>::NDArray(const std::vector<size_t>& shape, const T& init_val) : shape_(shape) {
    strides_ = compute_strides(shape_);
    size_t total = 1;
    for (auto s : shape_){
        if(s <= 0)
            throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    data_ = new T[total];
    fill(init_val);  // Use parallel fill to initialize
}

template<typename T>
NDArray<T>::NDArray(std::initializer_list<size_t> shape, const T& init_val)
    : shape_(shape), strides_(compute_strides(shape_)) {
    size_t total = 1;
    for (auto s : shape_){
        if(s <= 0)
            throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    data_ = new T[total];
    fill(init_val);  // Use parallel fill to initialize
}

//
// Basic Array Properties
//

template<typename T>
std::vector<size_t> NDArray<T>::shape() const {
    return shape_;
}

template<typename T>
size_t NDArray<T>::ndim() const {
    return shape_.size();
}

template<typename T>
size_t NDArray<T>::size() const {
    if (shape_.empty()) return 0;

    size_t total = 1;
    for (auto s : shape_)
        total *= s;
    return total;
}

template<typename T>
NDArray<T> NDArray<T>::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = 1;
    for (auto dim : new_shape)
        new_size *= dim;
    if (new_size != size())
        throw std::runtime_error("New shape is incompatible with array size");
    NDArray<T> new_array(*this);
    new_array.shape_ = new_shape;
    new_array.strides_ = new_array.compute_strides(new_shape);
    return new_array;
}

template<typename T>
std::vector<T> NDArray<T>::flatten() const {
    size_t total = size(); // Total number of elements
    std::vector<T> flat(total); // Vector to store the flattened array

    unsigned nthreads = std::thread::hardware_concurrency(); // Number of threads

    if (nthreads == 0) nthreads = 2; // Set to 2 if hardware_concurrency() returns 0
    size_t block = total / nthreads; // Number of elements per thread

    std::vector<std::thread> threads; // Vector to store threads
    for (unsigned i = 0; i < nthreads; i++) { // Loop over threads
        size_t start = i * block;               // Start index
        size_t end = (i == nthreads - 1) ? total : start + block; // End index
        threads.push_back(std::thread([this, &flat, start, end]() { // Explicitly capture start and end
            for (size_t j = start; j < end; j++) { // Loop over elements
                flat[j] = data_[j];                 // Copy element to flat array
            }
        }));
    }
    for (auto &t : threads) // Join threads
        t.join();   // Wait for thread to finish    
    return flat;    // Return the flattened array
}


template<typename T>
T NDArray<T>::sum() const {
    T total = 0;
    size_t total_size = size();
    unsigned nthreads = std::thread::hardware_concurrency(); // Number of threads

    if (nthreads == 0) nthreads = 2; // Set to 2 if hardware_concurrency() returns 0
    size_t block = total_size / nthreads; // Number of elements per thread

    std::vector<std::thread> threads; // Vector to store threads

    for (unsigned i = 0; i < nthreads; i++) { // Loop over threads
        size_t start = i * block;               // Start index
        size_t end = (i == nthreads - 1) ? total_size : start + block; // End index
        threads.push_back(std::thread([this, &total, start, end]() { // Explicitly capture start and end
            for (size_t j = start; j < end; j++) { // Loop over elements
                total += data_[j];                  // Add element to total
            }
        }));
    }
    for (auto &t : threads) // Join threads
        t.join();   // Wait for thread to finish
    return total;   // Return the sum
}


template<typename T>
T NDArray<T>::mean() const {
    return sum() / size(); // Compute the mean
}

template<typename T>
T NDArray<T>::min() const {
    T min_val = std::numeric_limits<T>::max(); // Initialize to maximum value
    size_t total_size = size();
    unsigned nthreads = std::thread::hardware_concurrency(); // Number of threads

    if (nthreads == 0) nthreads = 2; // Set to 2 if hardware_concurrency() returns 0
    size_t block = total_size / nthreads; // Number of elements per thread

    std::vector<std::thread> threads; // Vector to store threads

    for (unsigned i = 0; i < nthreads; i++) { // Loop over threads
        size_t start = i * block;               // Start index
        size_t end = (i == nthreads - 1) ? total_size : start + block; // End index
        threads.push_back(std::thread([this, &min_val, start, end]() { // Explicitly capture start and end by value
            for (size_t j = start; j < end; j++) { // Loop over elements
                if (data_[j] < min_val) {           // Check if element is less than min_val
                    min_val = data_[j];             // Update min_val
                }
            }
        }));
    }
    for (auto &t : threads) // Join threads
        t.join();   // Wait for thread to finish
    return min_val; // Return the minimum value
}

template<typename T>
T NDArray<T>::max() const {
    T max_val = std::numeric_limits<T>::lowest(); // Initialize to lowest value
    size_t total_size = size();
    unsigned nthreads = std::thread::hardware_concurrency(); // Number of threads

    if (nthreads == 0) nthreads = 2; // Set to 2 if hardware_concurrency() returns 0
    size_t block = total_size / nthreads; // Number of elements per thread

    std::vector<std::thread> threads; // Vector to store threads

    for (unsigned i = 0; i < nthreads; i++) { // Loop over threads
        size_t start = i * block;               // Start index
        size_t end = (i == nthreads - 1) ? total_size : start + block; // End index
        threads.push_back(std::thread([this, &max_val, start, end]() { // Explicitly capture start and end by value
            for (size_t j = start; j < end; j++) { // Loop over elements
                if (data_[j] > max_val) {           // Check if element is greater than max_val
                    max_val = data_[j];             // Update max_val
                }
            }
        }));
    }
    for (auto &t : threads) // Join threads
        t.join();   // Wait for thread to finish
    return max_val; // Return the maximum value
}


// Compute the determinant of the matrix
template<typename T>
T NDArray<T>::determinant() const {
    if (!is_square()) {
        throw std::runtime_error("Matrix must be square to compute determinant.");
    }

    if (shape_[0] == 1) {
        return data_[0];
    }

    if (shape_[0] == 2) {
        return data_[0] * data_[3] - data_[1] * data_[2];
    }

    return determinant_helper();
}

template<typename T>
bool NDArray<T>::is_square() const {
    return std::all_of(shape_.begin(), shape_.end(), [this](size_t s) { return s == shape_[0]; });
}

//
// Modify Array
//

template<typename T>
void NDArray<T>::fill(const T& value) {
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency(); // Number of threads

    if (nthreads == 0) nthreads = 2; // Set to 2 if hardware_concurrency() returns 0
    size_t block = total / nthreads; // Number of elements per thread

    std::vector<std::thread> threads; // Vector to store threads

    for (unsigned i = 0; i < nthreads; i++) { // Loop over threads
        size_t start = i * block;               // Start index
        size_t end = (i == nthreads - 1) ? total : start + block; // End index
        threads.push_back(std::thread([this, start, end, value]() { // Create a thread
            for (size_t j = start; j < end; j++) { // Loop over elements
                data_[j] = value;                   // Set element to value
            }
        }));
    }
    for (auto &t : threads) // Join threads
        t.join();       // Wait for thread to finish
}

template<typename T>
void NDArray<T>::zeros() {
    fill(T(0));     // Fill the array with zeros using the fill method
}

template<typename T>
void NDArray<T>::ones() {
    fill(T(1));    // Fill the array with ones using the fill method
}

template<typename T>
void NDArray<T>::transpose() {
    // This implementation reverses the axes.
    std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());  // Reverse the shape
    std::vector<size_t> new_strides = compute_strides(new_shape);   // Compute strides for the new shape
    size_t total = size();                                          // Total number of elements
    T* new_data = new T[total];                                     // Allocate memory for the new data
    unsigned nthreads = std::thread::hardware_concurrency();        // Number of threads
    if (nthreads == 0) nthreads = 2;                                // Set to 2 if hardware_concurrency() returns 0
    size_t block = total / nthreads;                                // Number of elements per thread
    std::vector<std::thread> threads;                               // Vector to store threads
    for (unsigned i = 0; i < nthreads; i++) {                       // Loop over threads
        size_t start = i * block;                                   // Start index
        size_t end = (i == nthreads - 1) ? total : start + block;   // End index
        threads.push_back(std::thread([=, &new_data, &new_shape, this]() { // Create a thread
            for (size_t j = start; j < end; j++) {
                // Convert flat index j into multi-index for the new shape.
                std::vector<size_t> new_idx(new_shape.size());
                size_t tmp = j;
                for (int k = static_cast<int>(new_shape.size()) - 1; k >= 0; k--) {
                    new_idx[k] = tmp % new_shape[k];
                    tmp /= new_shape[k];
                }
                // Reverse indices to obtain the corresponding indices in the original array.
                std::vector<size_t> orig_idx(new_idx.rbegin(), new_idx.rend());
                size_t orig_flat = compute_index(orig_idx);
                new_data[j] = data_[orig_flat];
            }
        }));
    }
    for (auto &t : threads)
        t.join();
    delete[] data_;
    data_ = new_data;
    shape_ = new_shape;
    strides_ = new_strides;
}

template<typename T>
void NDArray<T>::reverse() {
    // Reverse the flat data order.
    size_t total = size();
    size_t half = total / 2;
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = half / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? half : start + block;
        threads.push_back(std::thread([=, total, this]() {
            for (size_t j = start; j < end; j++) {
                std::swap(data_[j], data_[total - 1 - j]);
            }
        }));
    }
    for (auto &t : threads)
        t.join();
}

template<typename T>
void NDArray<T>::pow(const T& exponent) {
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;              // Start index
        size_t end = (i == nthreads - 1) ? total : start + block; // End index
        threads.push_back(std::thread([=, this]() {     // Create a thread
            for (size_t j = start; j < end; j++) { // Loop over elements
                data_[j] = std::pow(data_[j], exponent); // Compute the power
            }
        }));
    }
    for (auto &t : threads)
        t.join();
}

template <typename T>
void NDArray<T>::invert() {
    if (!is_square()) {
        throw std::invalid_argument("Matrix must be square to compute inverse.");
    }

    if (shape_[0] == 1) {
        data_[0] = 1 / data_[0];
        return;
    }

    if (shape_[0] == 2) {
        T det = data_[0] * data_[3] - data_[1] * data_[2];
        if (std::abs(det) < 1e-10) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
        T temp = data_[0];
        data_[0] = data_[3] / det;
        data_[1] = -data_[1] / det;
        data_[2] = -data_[2] / det;
        data_[3] = temp / det;
        return;
    }

    if (shape_.size() != 2) {
        throw std::invalid_argument("Matrix must be 2D to compute inverse.");
    }

    invert_helper();
}

template<typename T>
std::vector<size_t> NDArray<T>::compute_strides(const std::vector<size_t>& shape) const {
    std::vector<size_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template<typename T>
size_t NDArray<T>::compute_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size())
        throw std::runtime_error("Number of indices does not match array dimension");
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i])
            throw std::runtime_error("Index out of bounds");
        index += indices[i] * strides_[i];
    }
    return index;
}


//
// Element Access
//

template<typename T>
T& NDArray<T>::operator()(int index) {
    return data_[index];
}

template<typename T>
T& NDArray<T>::operator()(const std::vector<size_t>& indices) {
    return data_[compute_index(indices)];
}

template<typename T>
const T& NDArray<T>::operator()(const std::vector<size_t>& indices) const {
    return data_[compute_index(indices)];
}



//
// Return Modified Array (non in-place)
//

template<typename T>
NDArray<T> NDArray<T>::filled(const T& value) const {
    NDArray<T> result = copy();
    result.fill(value);
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::zeros_like() const {
    return filled(T(0));
}

template<typename T>
NDArray<T> NDArray<T>::ones_like() const {
    return filled(T(1));
}

template<typename T>
NDArray<T> NDArray<T>::transposed() const {
    NDArray<T> result = copy();
    result.transpose();
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::powed(const T& exponent) const {
    NDArray<T> result = copy();
    result.pow(exponent);
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::reversed() const {
    NDArray<T> result = copy();
    result.reverse();
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::inverted() const {
    NDArray<T> result = copy();
    result.invert();
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::kernel() const {
    NDArray<T> result = copy();
    result.kernel();
    return result;
}

//
// Copy utility
//

template<typename T>
NDArray<T> NDArray<T>::copy() const {
    NDArray<T> new_array;
    new_array.shape_ = shape_;
    new_array.strides_ = strides_;
    size_t total = size();
    new_array.data_ = new T[total];
    std::copy(data_, data_ + total, new_array.data_);
    return new_array;
}

//
// Arithmetic Operators (element-wise)
//

template<typename T>
NDArray<T> NDArray<T>::operator+(const NDArray<T>& other) const {
    if (shape_ != other.shape_)                                             // Check if shapes match
        throw std::runtime_error("Shapes do not match for addition");       // Throw error if shapes do not match
    NDArray<T> result(shape_);                                    // Create a new array with the same shape                         
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();                // Number of threads
    if (nthreads == 0) nthreads = 2;                                // Set to 2 if hardware_concurrency() returns 0
    size_t block = total / nthreads;                            // Number of elements per thread                    
    std::vector<std::thread> threads;                        // Vector to store threads                             
    for (unsigned i = 0; i < nthreads; i++) {             // Loop over threads      
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;           // End index
        threads.push_back(std::thread([=, this, &other, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] + other.data_[j];      // Add the elements
            }
        }));
    }
    for (auto &t : threads)
        t.join();
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::operator-(const NDArray<T>& other) const {
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for subtraction");            // Throw error if shapes do not match
    NDArray<T> result(shape_);                                                      // Create a new array with the same shape
    size_t total = size();                                                        // Total number of elements                   
    unsigned nthreads = std::thread::hardware_concurrency();                    // Number of threads        
    if (nthreads == 0) nthreads = 2;                                                // Set to 2 if hardware_concurrency() returns 0
    size_t block = total / nthreads;                                                // Number of elements per thread
    std::vector<std::thread> threads;                                            // Vector to store threads                 
    for (unsigned i = 0; i < nthreads; i++) {                                           // Loop over threads
        size_t start = i * block;   
        size_t end = (i == nthreads - 1) ? total : start + block;                // End index
        threads.push_back(std::thread([=, this, &other, &result]() {                // Create a thread
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] - other.data_[j];                  // Subtract the elements
            }
        }));
    }
    for (auto &t : threads)
        t.join();
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::operator*(const NDArray<T>& other) const {
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for multiplication");
    NDArray<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &other, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] * other.data_[j];
            }
        }));
    }
    for (auto &t : threads)
        t.join();
    return result;
}

template<typename T>
NDArray<T> NDArray<T>::operator/(const NDArray<T>& other) const {
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for division");
    NDArray<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &other, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] / other.data_[j];
            }
        }));
    }
    for (auto &t : threads)
        t.join();
    return result;
}

//
// Dot Product (Matrix Multiplication for 2D arrays)
//

template<typename T>
NDArray<T> NDArray<T>::dot(const NDArray<T>& other) const {
    if (ndim() != 2 || other.ndim() != 2)
        throw std::runtime_error("dot product is only implemented for 2D arrays");      // Throw error if not 2D arrays
    size_t m = shape_[0], n = shape_[1];                                               // Get the shape of the first array
    if (other.shape_[0] != n)                                                           // Check if shapes align for dot product
        throw std::runtime_error("Shapes do not align for dot product");            // Throw error if shapes do not align
    size_t p = other.shape_[1];                                                    // Get the shape of the second array
    NDArray<T> result({ m, p }, T(0));                                             // Create a new array with the shape of the dot product
    unsigned nthreads = std::thread::hardware_concurrency();                        // Number of threads
    if (nthreads == 0) nthreads = 2;
    size_t block = m / nthreads;
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < nthreads; t++) {                                // Loop over threads
        size_t row_start = t * block;                                         // Start index
        size_t row_end = (t == nthreads - 1) ? m : row_start + block;           // End index
        threads.push_back(std::thread([=, this, &other, &result]() {           // Create a thread
            for (size_t i = row_start; i < row_end; i++) {               // Loop over rows
                for (size_t j = 0; j < p; j++) {               // Loop over columns
                    T sum = T(0);                                               // Initialize sum
                    for (size_t k = 0; k < n; k++) {
                        // Using the element-access operator for clarity.
                        sum += this->operator()({ i, k }) * other.operator()({ k, j });                 // Compute the dot product
                    }
                    result({ i, j }) = sum;                                                     // Set the result
                }
            }
        }));
    }
    for (auto &t : threads)
        t.join();
    return result;
}

//
// Utility: Print the flattened array
//

template<typename T>
void NDArray<T>::print() const {
    std::cout << "[";
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        std::cout << data_[i];
        if (i != total - 1)
            std::cout << ", ";
    }
    std::cout << "]\n";
}

template <typename T>
T NDArray<T>::determinant_helper() const {
    if (!is_square()) {
        throw std::invalid_argument("Matrix must be square to compute determinant.");
    }

    size_t n = shape_[0];
    NDArray<T> mat(*this);  // Create a copy of the matrix
    T det = 1;

    // Perform Gaussian elimination (with partial pivoting)
    for (size_t i = 0; i < n; ++i) {
        size_t pivotRow = i;

        // Find the row with the largest pivot
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(mat({j, i})) > std::abs(mat({pivotRow, i}))) {
                pivotRow = j;
            }
        }

        // If the pivot is too small (near zero), the matrix is singular
        if (std::abs(mat({pivotRow, i})) < 1e-10) {
            return 0;  // The determinant is zero
        }

        if (i != pivotRow) {
            size_t row_size = shape_[1];
            T* start_i = mat.data_ + i * row_size;
            T* start_pivot = mat.data_ + pivotRow * row_size;
            std::swap_ranges(start_i, start_i + row_size, start_pivot);
            det = -det;  // Row swap changes the sign of the determinant
        }

        // Eliminate entries below the pivot
        for (size_t j = i + 1; j < n; ++j) {
            T factor = mat({j, i}) / mat({i, i});
            for (size_t k = i; k < n; ++k) {
                mat({j, k}) -= factor * mat({i, k});
            }
        }

        // Multiply the diagonal element into the determinant
        det *= mat({i, i});
    }

    return det;
}

template <typename T>
void NDArray<T>::invert_helper() {
    if (!is_square()) {
        throw std::invalid_argument("Matrix must be square to compute inverse.");
    }

    T det = determinant_helper();
    if (std::abs(det) < 1e-10) {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }

    size_t n = shape_[0];
    NDArray<T> mat(*this);  // Create a copy of the matrix
    NDArray<T> inv(n, n);   // Initialize the inverse matrix

    // Augment the matrix with the identity matrix
    for (size_t i = 0; i < n; ++i) {
        inv(i, i) = 1;
    }

    // Perform Gaussian elimination with row operations
    for (size_t i = 0; i < n; ++i) {
        // Find pivot row
        size_t pivotRow = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(mat(j, i)) > std::abs(mat(pivotRow, i))) {
                pivotRow = j;
            }
        }

        // Swap rows if necessary
        if (i != pivotRow) {
            std::swap(mat.data_[i], mat.data_[pivotRow]);
            std::swap(inv.data_[i], inv.data_[pivotRow]);
        }

        // Normalize the pivot row
        T pivot = mat(i, i);
        for (size_t j = 0; j < n; ++j) {
            mat(i, j) /= pivot;
            inv(i, j) /= pivot;
        }

        // Eliminate entries in other rows
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                T factor = mat(j, i);
                for (size_t k = 0; k < n; ++k) {
                    mat(j, k) -= factor * mat(i, k);
                    inv(j, k) -= factor * inv(i, k);
                }
            }
        }
    }

    // Copy the inverse into the original matrix (for in-place operation)
    *this = inv;
}


} // namespace NumCPP

#endif // NDARRAY_TPP


