#ifndef ARRAY_TPP
#define ARRAY_TPP

#include "Array.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <thread>

namespace NumCPP {

template<typename T>
Array<T>::Array() : data_(nullptr) {}

template<typename T>
Array<T>::~Array() {
    delete[] data_;
}

template<typename T>
Array<T>::Array(const Array<T>& other) : shape_(other.shape_), strides_(other.strides_) {
    size_t total = other.size();
    data_ = new T[total];
    std::copy(other.data_, other.data_ + total, data_);
}

template<typename T>
Array<T>::Array(Array<T>&& other) noexcept : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), data_(other.data_) {
    other.data_ = nullptr;
}

template<typename T>
Array<T>& Array<T>::operator=(const Array<T>& other) {
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
Array<T>& Array<T>::operator=(Array<T>&& other) noexcept {
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
Array<T>::Array(const std::vector<size_t>& shape, const T& init_val) : shape_(shape) {
    strides_ = compute_strides(shape_);
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0) throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    data_ = new T[total];
    fill(init_val);
}

template<typename T>
Array<T>::Array(std::initializer_list<size_t> shape, const T& init_val) : shape_(shape), strides_(compute_strides(shape_)) {
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0) throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    data_ = new T[total];
    fill(init_val);
}

template<typename T>
Array<T>::Array(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(shape), strides_(compute_strides(shape_)) {
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0) throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    if (data.size() != total) throw std::invalid_argument("Data size does not match shape");
    data_ = new T[total];
    std::copy(data.begin(), data.end(), data_);
}

template<typename T>
Array<T>::Array(std::initializer_list<size_t> shape, const std::vector<T>& data) : shape_(shape), strides_(compute_strides(shape_)) {
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0) throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    if (data.size() != total) throw std::invalid_argument("Data size does not match shape");
    data_ = new T[total];
    std::copy(data.begin(), data.end(), data_);
}

template<typename T>
std::vector<size_t> Array<T>::shape() const {
    return shape_;
}

template<typename T>
size_t Array<T>::ndim() const {
    return shape_.size();
}

template<typename T>
size_t Array<T>::size() const {
    if (shape_.empty()) return 0;
    size_t total = 1;
    for (auto s : shape_) total *= s;
    return total;
}

template<typename T>
std::vector<size_t> Array<T>::strides() const {
    return strides_;
}

template<typename T>
Array<T> Array<T>::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_size = 1;
    for (auto dim : new_shape) new_size *= dim;
    if (new_size != size()) throw std::runtime_error("New shape is incompatible with array size");
    Array<T> new_array(*this);
    new_array.shape_ = new_shape;
    new_array.strides_ = new_array.compute_strides(new_shape);
    return new_array;
}

template<typename T>
std::vector<T> Array<T>::flatten() const {
    size_t total = size();
    std::vector<T> flat(total);
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([this, &flat, start, end]() {
            for (size_t j = start; j < end; j++) {
                flat[j] = data_[j];
            }
        }));
    }
    for (auto &t : threads) t.join();
    return flat;
}

template<typename T>
T Array<T>::sum() const {
    T total = 0;
    size_t total_size = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total_size / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total_size : start + block;
        threads.push_back(std::thread([this, &total, start, end]() {
            for (size_t j = start; j < end; j++) {
                total += data_[j];
            }
        }));
    }
    for (auto &t : threads) t.join();
    return total;
}

template<typename T>
T Array<T>::mean() const {
    return sum() / size();
}

template<typename T>
T Array<T>::min() const {
    T min_val = std::numeric_limits<T>::max();
    size_t total_size = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total_size / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total_size : start + block;
        threads.push_back(std::thread([this, &min_val, start, end]() {
            for (size_t j = start; j < end; j++) {
                if (data_[j] < min_val) {
                    min_val = data_[j];
                }
            }
        }));
    }
    for (auto &t : threads) t.join();
    return min_val;
}

template<typename T>
T Array<T>::max() const {
    T max_val = std::numeric_limits<T>::lowest();
    size_t total_size = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total_size / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total_size : start + block;
        threads.push_back(std::thread([this, &max_val, start, end]() {
            for (size_t j = start; j < end; j++) {
                if (data_[j] > max_val) {
                    max_val = data_[j];
                }
            }
        }));
    }
    for (auto &t : threads) t.join();
    return max_val;
}

template<typename T>
bool Array<T>::is_square() const {
    return std::all_of(shape_.begin(), shape_.end(), [this](size_t s) { return s == shape_[0]; });
}

template<typename T>
void Array<T>::fill(const T& value) {
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([this, start, end, value]() {
            for (size_t j = start; j < end; j++) {
                data_[j] = value;
            }
        }));
    }
    for (auto &t : threads) t.join();
}

template<typename T>
void Array<T>::zeros() {
    fill(T(0));
}

template<typename T>
void Array<T>::ones() {
    fill(T(1));
}

template<typename T>
void Array<T>::transpose() {
    std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());
    std::vector<size_t> new_strides = compute_strides(new_shape);
    size_t total = size();
    T* new_data = new T[total];
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=]() {
            for (size_t j = start; j < end; j++) {
                std::vector<size_t> new_idx(new_shape.size());
                size_t tmp = j;
                for (int k = static_cast<int>(new_shape.size()) - 1; k >= 0; k--) {
                    new_idx[k] = tmp % new_shape[k];
                    tmp /= new_shape[k];
                }
                std::vector<size_t> orig_idx(new_idx.rbegin(), new_idx.rend());
                size_t orig_flat = 0;
                for (size_t k = 0; k < orig_idx.size(); k++) {
                    orig_flat += orig_idx[k] * strides_[k];
                }
                new_data[j] = data_[orig_flat];
            }
        }));
    }
    for (auto &t : threads) t.join();
    delete[] data_;
    data_ = new_data;
    shape_ = new_shape;
    strides_ = new_strides;
}

template<typename T>
void Array<T>::reverse() {
    size_t total = size();
    T* new_data = new T[total];
    for (size_t i = 0; i < total; i++) {
        new_data[i] = data_[total - 1 - i];
    }
    delete[] data_;
    data_ = new_data;
}

template<typename T>
void Array<T>::pow(const T& exponent) {
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([this, start, end, exponent]() {
            for (size_t j = start; j < end; j++) {
                data_[j] = std::pow(data_[j], exponent);
            }
        }));
    }
    for (auto &t : threads) t.join();
}

template<typename T>
Array<T> Array<T>::copy() const {
    Array<T> new_array;
    new_array.shape_ = shape_;
    new_array.strides_ = strides_;
    size_t total = size();
    new_array.data_ = new T[total];
    std::copy(data_, data_ + total, new_array.data_);
    return new_array;
}

template<typename T>
T& Array<T>::operator()(size_t index) {
    return data_[index];
}

template<typename T>
const T& Array<T>::operator()(size_t index) const {
    return data_[index];
}

template<typename T>
T& Array<T>::operator()(const std::vector<size_t>& indices) {
    return data_[compute_index(indices)];
}

template<typename T>
const T& Array<T>::operator()(const std::vector<size_t>& indices) const {
    return data_[compute_index(indices)];
}

template<typename T>
T& Array<T>::operator[](size_t index) {
    return operator()({index});
}

template<typename T>
const T& Array<T>::operator[](size_t index) const {
    return operator()({index});
}

template<typename T>
T& Array<T>::operator[](const std::vector<size_t>& indices) {
    return operator()(indices);
}

template<typename T>
const T& Array<T>::operator[](const std::vector<size_t>& indices) const {
    return operator()(indices);
}

template<typename T>
Array<T> Array<T>::filled(const T& value) const {
    Array<T> result = copy();
    result.fill(value);
    return result;
}

template<typename T>
Array<T> Array<T>::zeros_like() const {
    return filled(T(0));
}

template<typename T>
Array<T> Array<T>::ones_like() const {
    return filled(T(1));
}

template<typename T>
Array<T> Array<T>::transposed() const {
    Array<T> result = copy();
    result.transpose();
    return result;
}

template<typename T>
Array<T> Array<T>::powed(const T& exponent) const {
    Array<T> result = copy();
    result.pow(exponent);
    return result;
}

template<typename T>
Array<T> Array<T>::reversed() const {
    Array<T> result = copy();
    result.reverse();
    return result;
}

template<typename T>
Array<T> Array<T>::operator+(const Array<T>& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shapes do not match for addition");
    Array<T> result(shape_);
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
                result.data_[j] = this->data_[j] + other.data_[j];
            }
        }));
    }
    for (auto &t : threads) t.join();
    return result;
}

template<typename T>
Array<T> Array<T>::operator-(const Array<T>& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shapes do not match for subtraction");
    Array<T> result(shape_);
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
                result.data_[j] = this->data_[j] - other.data_[j];
            }
        }));
    }
    for (auto &t : threads) t.join();
    return result;
}

template<typename T>
Array<T> Array<T>::operator*(const Array<T>& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shapes do not match for multiplication");
    Array<T> result(shape_);
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
    for (auto &t : threads) t.join();
    return result;
}

template<typename T>
Array<T> Array<T>::operator/(const Array<T>& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shapes do not match for division");
    Array<T> result(shape_);
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
    for (auto &t : threads) t.join();
    return result;
}

template<typename T>
void Array<T>::print() const {
    std::cout << "[";
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        std::cout << data_[i];
        if (i != total - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

template<typename T>
std::vector<size_t> Array<T>::compute_strides(const std::vector<size_t>& shape) const {
    std::vector<size_t> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template<typename T>
size_t Array<T>::compute_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) throw std::runtime_error("Number of indices does not match array dimension");
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i]) throw std::runtime_error("Index out of bounds");
        index += indices[i] * strides_[i];
    }
    return index;
}

} // namespace NumCPP

#endif // ARRAY_TPP