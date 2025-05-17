#ifndef ARRAY_TPP
#define ARRAY_TPP

#include "Array.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace NumCPP {

template <typename T>
Array<T>::Array()
    : shape_()
    , strides_()
    , data_(nullptr)
{
}

template <typename T>
Array<T>::~Array()
{
    delete[] data_;
}

template <typename T>
Array<T>::Array(const Array<T>& other)
    : shape_(other.shape_)
    , strides_(other.strides_)
{
    size_t total = other.size();
    data_ = new T[total];
    std::copy(other.data_, other.data_ + total, data_);
}

template <typename T>
Array<T>::Array(Array<T>&& other) noexcept
    : shape_(std::move(other.shape_))
    , strides_(std::move(other.strides_))
    , data_(other.data_)
{
    other.data_ = nullptr;
}

template <typename T>
Array<T>& Array<T>::operator=(const Array<T>& other)
{
    using std::swap;
    Array<T> temp(other);
    swap(shape_, temp.shape_);
    swap(strides_, temp.strides_);
    swap(data_, temp.data_);
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator=(Array<T>&& other) noexcept
{
    if (this != &other) {
        delete[] data_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        data_ = other.data_;
        other.data_ = nullptr;
    }
    return *this;
}

template <typename T>
Array<T>::Array(const std::vector<size_t>& shape, const T& init_val)
    : shape_(shape)
{
    strides_ = compute_strides(shape_);
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0)
            throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    data_ = new T[total];
    fill(init_val);
}

template <typename T>
Array<T>::Array(std::initializer_list<size_t> shape, const T& init_val)
    : shape_(shape)
    , strides_(compute_strides(shape_))
{
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0)
            throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    data_ = new T[total];
    fill(init_val);
}

template <typename T>
Array<T>::Array(const std::vector<size_t>& shape, const std::vector<T>& data)
    : shape_(shape)
    , strides_(compute_strides(shape_))
{
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0)
            throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    if (data.size() != total)
        throw std::invalid_argument("Data size does not match shape");
    data_ = new T[total];
    std::copy(data.begin(), data.end(), data_);
}

template <typename T>
Array<T>::Array(std::initializer_list<size_t> shape, const std::vector<T>& data)
    : shape_(shape)
    , strides_(compute_strides(shape_))
{
    size_t total = 1;
    for (auto s : shape_) {
        if (s <= 0)
            throw std::invalid_argument("Shape dimensions must be positive");
        total *= s;
    }
    if (data.size() != total)
        throw std::invalid_argument("Data size does not match shape");
    data_ = new T[total];
    std::copy(data.begin(), data.end(), data_);
}

template <typename T>
std::vector<size_t> Array<T>::shape() const
{
    return shape_;
}

template <typename T>
size_t Array<T>::ndim() const
{
    return shape_.size();
}

template <typename T>
size_t Array<T>::size() const
{
    if (shape_.empty())
        return 0;
    size_t total = 1;
    for (auto s : shape_)
        total *= s;
    return total;
}

template <typename T>
std::vector<size_t> Array<T>::strides() const
{
    return strides_;
}

template <typename T>
Array<T> Array<T>::reshape(const std::vector<size_t>& new_shape) const
{
    Array<T> new_array(*this);
    new_array.shape_ = new_shape;
    new_array.strides_ = new_array.compute_strides(new_shape);
    return new_array;
}

template <typename T>
std::vector<T> Array<T>::flatten() const
{
    size_t total = size();
    std::vector<T> flat(total);
    if (total < 1000) {
        std::copy(data_, data_ + total, flat.begin());
        return flat;
    }
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
    return flat;
}

template <typename T>
T Array<T>::sum() const
{
    T total = 0;
    size_t total_size = size();
    if (total_size < 1000) {
        for (size_t i = 0; i < total_size; i++) {
            total += data_[i];
        }
        return total;
    }

    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total_size / nthreads;
    std::vector<std::thread> threads;
    std::vector<T> partial_sums(nthreads, T(0));
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total_size : start + block;
        threads.push_back(std::thread([this, start, end, &partial_sums, i]() {
            T local_sum = T(0);
            for (size_t j = start; j < end; j++) {
                local_sum += data_[j];
            }
            partial_sums[i] = local_sum;
        }));
    }
    for (auto& t : threads)
        t.join();
    for (const auto& ps : partial_sums)
        total += ps;
    return total;
}

template <typename T>
T Array<T>::mean() const
{
    if (size() == 0)
        throw std::runtime_error("Cannot compute mean of empty array");
    return sum() / size();
}

template <typename T>
T Array<T>::min() const
{
    T min_val = std::numeric_limits<T>::max();
    size_t total_size = size();
    if (total_size < 1000) {
        for (size_t i = 0; i < total_size; i++) {
            if (data_[i] < min_val) {
                min_val = data_[i];
            }
        }
        return min_val;
    }

    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total_size / nthreads;
    std::vector<std::thread> threads;
    std::vector<T> thread_mins(nthreads, std::numeric_limits<T>::max());
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total_size : start + block;
        threads.push_back(std::thread([this, start, end, &thread_mins, i]() {
            T local_min = std::numeric_limits<T>::max();
            for (size_t j = start; j < end; j++) {
                if (data_[j] < local_min)
                    local_min = data_[j];
            }
            thread_mins[i] = local_min;
        }));
    }
    for (auto& t : threads)
        t.join();
    if (total_size == 0)
        throw std::runtime_error("Cannot compute min of empty array");
    min_val = thread_mins[0];
    for (const auto& tm : thread_mins)
        if (tm < min_val)
            min_val = tm;
    return min_val;
}

template <typename T>
T Array<T>::max() const
{
    T max_val = std::numeric_limits<T>::lowest();
    size_t total_size = size();
    if (total_size == 0)
        throw std::runtime_error("Cannot compute max of empty array");
    if (total_size < 1000) {
        for (size_t i = 0; i < total_size; i++) {
            if (data_[i] > max_val) {
                max_val = data_[i];
            }
        }
        return max_val;
    }
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total_size / nthreads;
    std::vector<std::thread> threads;
    std::vector<T> thread_maxs(nthreads, std::numeric_limits<T>::lowest());
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total_size : start + block;
        threads.push_back(std::thread([this, start, end, &thread_maxs, i]() {
            T local_max = std::numeric_limits<T>::lowest();
            for (size_t j = start; j < end; j++) {
                if (data_[j] > local_max)
                    local_max = data_[j];
            }
            thread_maxs[i] = local_max;
        }));
    }
    for (auto& t : threads)
        t.join();
    if (total_size == 0)
        throw std::runtime_error("Cannot compute max of empty array");
    max_val = thread_maxs[0];
    for (const auto& tm : thread_maxs)
        if (tm > max_val)
            max_val = tm;
    return max_val;
}

template <typename T>
bool Array<T>::is_square() const
{
    return std::all_of(shape_.begin(), shape_.end(), [this](size_t s) { return s == shape_[0]; });
}

template <typename T>
void Array<T>::fill(const T& value)
{
    size_t total = size();
    if (total < 1000) {
        for (size_t i = 0; i < total; i++) {
            data_[i] = value;
        }
        return;
    }
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
}

template <typename T>
void Array<T>::zeros()
{
    fill(T(0));
}

template <typename T>
void Array<T>::ones()
{
    fill(T(1));
}

template <typename T>
void Array<T>::transpose()
{
    std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());
    std::vector<size_t> new_strides = compute_strides(new_shape);
    size_t total = size();
    T* new_data = new T[total];
    if (total < 1000) {
        for (size_t i = 0; i < total; i++) {
            std::vector<size_t> new_idx(new_shape.size());
            size_t tmp = i;
            for (int k = static_cast<int>(new_shape.size()) - 1; k >= 0; k--) {
                new_idx[k] = tmp % new_shape[k];
                tmp /= new_shape[k];
            }
            std::vector<size_t> orig_idx(new_idx.rbegin(), new_idx.rend());
            size_t orig_flat = 0;
            for (size_t k = 0; k < orig_idx.size(); k++) {
                orig_flat += orig_idx[k] * strides_[k];
            }
            new_data[i] = data_[orig_flat];
        }
        delete[] data_;
        data_ = new_data;
        shape_ = new_shape;
        strides_ = new_strides;
        return;
    }
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
    delete[] data_;
    data_ = new_data;
    shape_ = new_shape;
    strides_ = new_strides;
}

template <typename T>
void Array<T>::reverse()
{
    if (shape_.size() == 0)
        return;
    size_t total = size();
    std::reverse(data_, data_ + total);
}

template <typename T>
void Array<T>::pow(const T& exponent)
{
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
}

template <typename T>
Array<T> Array<T>::copy() const
{
    Array<T> new_array;
    new_array.shape_ = shape_;
    new_array.strides_ = strides_;
    size_t total = size();
    new_array.data_ = new T[total];
    std::copy(data_, data_ + total, new_array.data_);
    return new_array;
}

template <typename T>
T& Array<T>::operator()(size_t index)
{
    if (index >= size() || index < 0)
        throw std::out_of_range("Index out of range");
    return data_[index];
}

template <typename T>
const T& Array<T>::operator()(size_t index) const
{
    if (index >= size() || index < 0)
        throw std::out_of_range("Index out of range");
    return data_[index];
}

template <typename T>
T& Array<T>::operator()(const std::vector<size_t>& indices)
{
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Number of indices must match number of dimensions");
    if (indices.size() == 0)
        throw std::invalid_argument("No indices provided");
    size_t index = compute_index(indices);
    if (index >= size())
        throw std::out_of_range("Index out of range");
    return data_[index];
}

template <typename T>
const T& Array<T>::operator()(const std::vector<size_t>& indices) const
{
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Number of indices must match number of dimensions");
    if (indices.size() == 0)
        throw std::invalid_argument("No indices provided");
    size_t index = compute_index(indices);
    if (index >= size())
        throw std::out_of_range("Index out of range");
    return data_[index];
}

template <typename T>
template <typename... Indices>
T& Array<T>::operator()(Indices... indices)
{
    if (sizeof...(indices) != shape_.size())
        throw std::invalid_argument("Number of indices must match number of dimensions");
    if (sizeof...(indices) == 0)
        throw std::invalid_argument("No indices provided");
    if (sizeof...(indices) > shape_.size())
        throw std::invalid_argument("Too many indices provided");
    std::vector<size_t> idx = { static_cast<size_t>(indices)... };
    size_t index = compute_index(idx);
    if (index >= size())
        throw std::out_of_range("Index out of range");
    return data_[index];
}

template <typename T>
template <typename... Indices>
const T& Array<T>::operator()(Indices... indices) const
{
    std::vector<size_t> idx = { static_cast<size_t>(indices)... };
    if (sizeof...(indices) != shape_.size())
        throw std::invalid_argument("Number of indices must match number of dimensions");
    if (sizeof...(indices) == 0)
        throw std::invalid_argument("No indices provided");
    if (sizeof...(indices) > shape_.size())
        throw std::invalid_argument("Too many indices provided");
    size_t index = compute_index(idx);
    if (index >= size())
        throw std::out_of_range("Index out of range");
    return data_[compute_index(index)];
}

template <typename T>
T& Array<T>::operator[](size_t index)
{
    if (index >= size())
        throw std::out_of_range("Index out of range");
    return operator()({ index });
}

template <typename T>
const T& Array<T>::operator[](size_t index) const
{
    if (index >= size())
        throw std::out_of_range("Index out of range");
    return operator()({ index });
}

template <typename T>
T& Array<T>::operator[](const std::vector<size_t>& indices)
{
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Number of indices must match number of dimensions");
    return operator()(indices);
}

template <typename T>
const T& Array<T>::operator[](const std::vector<size_t>& indices) const
{
    if (indices.size() != shape_.size())
        throw std::invalid_argument("Number of indices must match number of dimensions");
    return operator()(indices);
}

template <typename T>
Array<T> Array<T>::filled(const T& value) const
{
    Array<T> result = copy();
    result.fill(value);
    return result;
}

template <typename T>
Array<T> Array<T>::zeros_like() const
{
    return filled(T(0));
}

template <typename T>
Array<T> Array<T>::ones_like() const
{
    return filled(T(1));
}

template <typename T>
Array<T> Array<T>::transposed() const
{
    Array<T> result = copy();
    result.transpose();
    return result;
}

template <typename T>
Array<T> Array<T>::powed(const T& exponent) const
{
    Array<T> result = copy();
    result.pow(exponent);
    return result;
}

template <typename T>
Array<T> Array<T>::reversed() const
{
    Array<T> result = copy();
    result.reverse();
    return result;
}

template <typename T>
Array<T> Array<T>::operator+(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for addition");
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator-(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for subtraction");
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator*(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for multiplication");
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator/(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for division");
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
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
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator+(const T& scalar) const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([&, this, scalar]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] + scalar;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator-(const T& scalar) const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([&, this, scalar]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] - scalar;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator*(const T& scalar) const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([&, this, scalar]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] * scalar;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator/(const T& scalar) const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([&, this, scalar]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j] / scalar;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T>& Array<T>::operator+=(const Array<T>& other)
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for addition");
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &other]() {
            for (size_t j = start; j < end; j++) {
                this->data_[j] += other.data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator-=(const Array<T>& other)
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for subtraction");
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &other]() {
            for (size_t j = start; j < end; j++) {
                this->data_[j] -= other.data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator*=(const Array<T>& other)
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for multiplication");
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &other]() {
            for (size_t j = start; j < end; j++) {
                this->data_[j] *= other.data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator/=(const Array<T>& other)
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for division");
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &other]() {
            for (size_t j = start; j < end; j++) {
                this->data_[j] /= other.data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator+=(const T& scalar)
{
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &scalar]() {
            for (size_t j = start; j < end; j++) {
                this->data_[j] += scalar;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator-=(const T& scalar)
{
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([&, this, scalar]() {
            for (size_t j = start; j < end; j++) {
                this->data_[j] -= scalar;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator*=(const T& scalar)
{
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this]() {
            for (size_t j = start; j < end; j++) {
                this->data_[j] *= scalar;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator/=(const T& scalar)
{
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this](size_t start, size_t end, T scalar) {
            for (size_t j = start; j < end; j++) {
                this->data_[j] /= scalar;
            }
        },
            start, end, scalar));
    }
    for (auto& t : threads)
        t.join();
    return *this;
}

template <typename T>
Array<T> Array<T>::operator-() const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = -this->data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator+() const
{
    return copy();
}

template <typename T>
Array<T> Array<T>::operator++()
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = ++this->data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator--()
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = --this->data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator++(int)
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j]++;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator--(int)
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j]--;
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator!() const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = !this->data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator~() const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = ~this->data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator&() const
{
    Array<T> result(shape_);
    size_t total = size();
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0)
        nthreads = 2;
    size_t block = total / nthreads;
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < nthreads; i++) {
        size_t start = i * block;
        size_t end = (i == nthreads - 1) ? total : start + block;
        threads.push_back(std::thread([=, this, &result]() {
            for (size_t j = start; j < end; j++) {
                result.data_[j] = this->data_[j];
            }
        }));
    }
    for (auto& t : threads)
        t.join();
    return result;
}

template <typename T>
Array<T> Array<T>::operator|(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for bitwise OR");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] | other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator^(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for bitwise XOR");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] ^ other.data_[i];
    }
    return result;
}

template <typename T>
Array<T>& Array<T>::operator&=(const Array<T>& other)
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for bitwise AND assignment");
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        data_[i] &= other.data_[i];
    }
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator|=(const Array<T>& other)
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for bitwise OR assignment");
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        data_[i] |= other.data_[i];
    }
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator^=(const Array<T>& other)
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for bitwise XOR assignment");
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        data_[i] ^= other.data_[i];
    }
    return *this;
}

template <typename T>
Array<T> Array<T>::operator&(const T& scalar) const
{
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] & scalar;
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator|(const T& scalar) const
{
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] | scalar;
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator^(const T& scalar) const
{
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] ^ scalar;
    }
    return result;
}

template <typename T>
Array<T>& Array<T>::operator&=(const T& scalar)
{
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        data_[i] &= scalar;
    }
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator|=(const T& scalar)
{
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        data_[i] |= scalar;
    }
    return *this;
}

template <typename T>
Array<T>& Array<T>::operator^=(const T& scalar)
{
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        data_[i] ^= scalar;
    }
    return *this;
}

template <typename T>
Array<T> Array<T>::operator==(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for equality comparison");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] == other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator!=(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for inequality comparison");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] != other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator<(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for less-than comparison");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] < other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator<=(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for less-than-or-equal comparison");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] <= other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator>(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for greater-than comparison");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] > other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator>=(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for greater-than-or-equal comparison");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] >= other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator&&(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for logical AND");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] && other.data_[i];
    }
    return result;
}

template <typename T>
Array<T> Array<T>::operator||(const Array<T>& other) const
{
    if (shape_ != other.shape_)
        throw std::runtime_error("Shapes do not match for logical OR");
    Array<T> result(shape_);
    size_t total = size();
    for (size_t i = 0; i < total; i++) {
        result.data_[i] = data_[i] || other.data_[i];
    }
    return result;
}

template <typename T>
void Array<T>::print() const
{
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
std::vector<size_t> Array<T>::compute_strides(const std::vector<size_t>& shape) const
{
    std::vector<size_t> strides(shape.size());
    if (shape.empty())
        return strides;
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

template <typename T>
size_t Array<T>::compute_index(const std::vector<size_t>& indices) const
{
    if (indices.size() != shape_.size())
        throw std::runtime_error("Number of indices does not match array dimension");
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] >= shape_[i])
            throw std::runtime_error("Index out of bounds");
        if (indices[i] < 0)
            throw std::runtime_error("Negative index not allowed");
        index += indices[i] * strides_[i];
    }
    return index;
}

} // namespace NumCPP

#endif // ARRAY_TPP