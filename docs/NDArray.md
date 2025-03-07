# NDArray Class Documentation

The `NDArray` class is a multi-dimensional array implementation in C++ designed for efficient numerical computations. It is part of the `NumCPP` namespace and supports operations such as element-wise arithmetic, matrix multiplication, and array transformations like transposition and inversion. The class is templated, allowing it to work with various data types (e.g., `double`, `int`), with `double` as the default.

## Table of Contents

1. [Constructors and Destructor](#constructors-and-destructor)
2. [Basic Array Properties](#basic-array-properties)
3. [Modify Array (In-Place)](#modify-array-in-place)
4. [Return Modified Array (Non In-Place)](#return-modified-array-non-in-place)
5. [Copy Utility](#copy-utility)
6. [Element Access](#element-access)
7. [Arithmetic Operators (Element-Wise)](#arithmetic-operators-element-wise)
8. [Dot Product (Matrix Multiplication)](#dot-product-matrix-multiplication)
9. [Utility](#utility)
10. [Private Helper Functions](#private-helper-functions)

---

## Constructors and Destructor

### `NDArray()`
- **Description**: Default constructor that creates an empty array.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr;
  ```

### `~NDArray()`
- **Description**: Destructor that deallocates the memory used by the array.

### `NDArray(const NDArray<T>& other)`
- **Description**: Copy constructor that creates a deep copy of another `NDArray`.
- **Parameters**:
  - `other`: The `NDArray` to copy.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr1({2, 2}, 1.0);
  NumCPP::NDArray<double> arr2(arr1);
  ```

### `NDArray(NDArray<T>&& other) noexcept`
- **Description**: Move constructor that transfers ownership of the data from another `NDArray`.
- **Parameters**:
  - `other`: The `NDArray` to move from.

### `NDArray<T>& operator=(const NDArray<T>& other)`
- **Description**: Copy assignment operator that creates a deep copy of another `NDArray`.
- **Parameters**:
  - `other`: The `NDArray` to copy.
- **Returns**: A reference to the assigned `NDArray`.

### `NDArray<T>& operator=(NDArray<T>&& other) noexcept`
- **Description**: Move assignment operator that transfers ownership of the data from another `NDArray`.
- **Parameters**:
  - `other`: The `NDArray` to move from.
- **Returns**: A reference to the assigned `NDArray`.

### `NDArray(const std::vector<size_t>& shape, const T& init_val = T())`
- **Description**: Constructor that creates an array with the specified shape and initializes all elements to `init_val`.
- **Parameters**:
  - `shape`: A vector representing the dimensions of the array.
  - `init_val`: The initial value for all elements (default is `T()`).
- **Throws**:
  - `std::invalid_argument` if any shape dimension is not positive.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr({2, 3}, 0.0);
  ```

### `NDArray(std::initializer_list<size_t> shape, const T& init_val = T())`
- **Description**: Constructor that creates an array with the specified shape (using an initializer list) and initializes all elements to `init_val`.
- **Parameters**:
  - `shape`: An initializer list representing the dimensions of the array.
  - `init_val`: The initial value for all elements (default is `T()`).
- **Throws**:
  - `std::invalid_argument` if any shape dimension is not positive.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr({2, 3}, 1.0);
  ```

### `NDArray(const std::vector<size_t>& shape, const std::vector<T>& data)`
- **Description**: Constructor that creates an array with the specified shape and copies data from a vector.
- **Parameters**:
  - `shape`: A vector representing the dimensions of the array.
  - `data`: A vector containing the data to copy into the array.
- **Throws**:
  - `std::invalid_argument` if any shape dimension is not positive or if the data size does not match the shape.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
  ```

### `NDArray(std::initializer_list<size_t> shape, const std::vector<T>& data)`
- **Description**: Constructor that creates an array with the specified shape (using an initializer list) and copies data from a vector.
- **Parameters**:
  - `shape`: An initializer list representing the dimensions of the array.
  - `data`: A vector containing the data to copy into the array.
- **Throws**:
  - `std::invalid_argument` if any shape dimension is not positive or if the data size does not match the shape.

---

## Basic Array Properties

### `std::vector<size_t> shape() const`
- **Description**: Returns the shape of the array.
- **Returns**: A vector representing the dimensions of the array.

### `size_t ndim() const`
- **Description**: Returns the number of dimensions of the array.
- **Returns**: The number of dimensions.

### `size_t size() const`
- **Description**: Returns the total number of elements in the array.
- **Returns**: The total number of elements.

### `std::vector<size_t> strides() const`
- **Description**: Returns the strides of the array, which indicate the number of elements to skip to move to the next position along each dimension.
- **Returns**: A vector representing the strides.

### `NDArray<T> reshape(const std::vector<size_t>& new_shape) const`
- **Description**: Returns a new `NDArray` with the specified shape, but the same data.
- **Parameters**:
  - `new_shape`: The new shape for the array.
- **Throws**:
  - `std::runtime_error` if the new shape is incompatible with the array size.
- **Returns**: A new `NDArray` with the specified shape.

### `std::vector<T> flatten() const`
- **Description**: Returns a flattened 1D vector of the array's data using multi-threading for efficiency.
- **Returns**: A vector containing all elements of the array in row-major order.

### `T sum() const`
- **Description**: Computes the sum of all elements in the array using multi-threading.
- **Returns**: The sum of the elements.

### `T mean() const`
- **Description**: Computes the mean (average) of all elements in the array.
- **Returns**: The mean of the elements.

### `T min() const`
- **Description**: Finds the minimum value in the array using multi-threading.
- **Returns**: The minimum value.

### `T max() const`
- **Description**: Finds the maximum value in the array using multi-threading.
- **Returns**: The maximum value.

### `T determinant() const`
- **Description**: Computes the determinant of the matrix (only for square matrices).
- **Throws**:
  - `std::runtime_error` if the matrix is not square.
- **Returns**: The determinant of the matrix.

### `bool is_square() const`
- **Description**: Checks if the array is a square matrix (i.e., all dimensions are equal).
- **Returns**: `true` if the array is square, `false` otherwise.

---

## Modify Array (In-Place)

These methods modify the array in place.

### `void fill(const T& value)`
- **Description**: Fills the entire array with the specified value using multi-threading.
- **Parameters**:
  - `value`: The value to fill the array with.

### `void zeros()`
- **Description**: Sets all elements of the array to zero.

### `void ones()`
- **Description**: Sets all elements of the array to one.

### `void transpose()`
- **Description**: Transposes the array (reverses the order of axes) using multi-threading.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  arr.transpose(); // Shape becomes {3, 2}
  ```

### `void reverse()`
- **Description**: Reverses the order of elements in the array.

### `void pow(const T& exponent)`
- **Description**: Raises each element of the array to the specified exponent using multi-threading.
- **Parameters**:
  - `exponent`: The exponent to raise each element to.

### `void invert()`
- **Description**: Computes the inverse of the matrix in place (only for square matrices).
- **Throws**:
  - `std::invalid_argument` if the matrix is not square or not 2D (for larger matrices).
  - `std::runtime_error` if the matrix is singular.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr({2, 2}, {4.0, 7.0, 2.0, 6.0});
  arr.invert();
  ```

---

## Return Modified Array (Non In-Place)

These methods return a new array with the modification applied, leaving the original array unchanged.

### `NDArray<T> filled(const T& value) const`
- **Description**: Returns a new array filled with the specified value.
- **Parameters**:
  - `value`: The value to fill the new array with.
- **Returns**: A new `NDArray` filled with `value`.

### `NDArray<T> zeros_like() const`
- **Description**: Returns a new array of the same shape filled with zeros.
- **Returns**: A new `NDArray` filled with zeros.

### `NDArray<T> ones_like() const`
- **Description**: Returns a new array of the same shape filled with ones.
- **Returns**: A new `NDArray` filled with ones.

### `NDArray<T> transposed() const`
- **Description**: Returns a new array that is the transpose of the original.
- **Returns**: A new `NDArray` that is the transpose.

### `NDArray<T> powed(const T& exponent) const`
- **Description**: Returns a new array with each element raised to the specified exponent.
- **Parameters**:
  - `exponent`: The exponent to raise each element to.
- **Returns**: A new `NDArray` with elements raised to the exponent.

### `NDArray<T> reversed() const`
- **Description**: Returns a new array with elements in reversed order.
- **Returns**: A new `NDArray` with reversed elements.

### `NDArray<T> inverted() const`
- **Description**: Returns a new array that is the inverse of the original matrix.
- **Throws**:
  - `std::invalid_argument` if the matrix is not square or not 2D (for larger matrices).
  - `std::runtime_error` if the matrix is singular.
- **Returns**: A new `NDArray` that is the inverse.

### `NDArray<T> kernel() const`
- **Description**: Placeholder method intended to return a kernel array (not implemented in the provided code).
- **Returns**: A new `NDArray` (currently a copy of the original).

---

## Copy Utility

### `NDArray<T> copy() const`
- **Description**: Creates and returns a deep copy of the array.
- **Returns**: A new `NDArray` that is a copy of the original.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr1({2, 2}, 1.0);
  NumCPP::NDArray<double> arr2 = arr1.copy();
  ```

---

## Element Access

### `T& operator()(size_t index)`
- **Description**: Accesses the element at the specified flat index.
- **Parameters**:
  - `index`: The flat index of the element.
- **Returns**: A reference to the element.

### `const T& operator()(size_t index) const`
- **Description**: Const version of flat index access.
- **Parameters**:
  - `index`: The flat index of the element.
- **Returns**: A const reference to the element.

### `T& operator()(const std::vector<size_t>& indices)`
- **Description**: Accesses the element at the specified multi-dimensional indices.
- **Parameters**:
  - `indices`: A vector of indices for each dimension.
- **Throws**:
  - `std::runtime_error` if the number of indices does not match the array dimensions or if an index is out of bounds.
- **Returns**: A reference to the element.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
  arr({1, 1}) = 5.0; // Sets element at position (1, 1) to 5.0
  ```

### `const T& operator()(const std::vector<size_t>& indices) const`
- **Description**: Const version of multi-dimensional index access.
- **Parameters**:
  - `indices`: A vector of indices for each dimension.
- **Throws**:
  - `std::runtime_error` if the number of indices does not match the array dimensions or if an index is out of bounds.
- **Returns**: A const reference to the element.

### `T& operator[](size_t index)`
- **Description**: Accesses the element at the specified flat index (alias for `operator()(index)`).
- **Parameters**:
  - `index`: The flat index of the element.
- **Returns**: A reference to the element.

### `const T& operator[](size_t index) const`
- **Description**: Const version of flat index access.
- **Parameters**:
  - `index`: The flat index of the element.
- **Returns**: A const reference to the element.

### `T& operator[](const std::vector<size_t>& indices)`
- **Description**: Accesses the element at the specified multi-dimensional indices (alias for `operator()(indices)`).
- **Parameters**:
  - `indices`: A vector of indices for each dimension.
- **Returns**: A reference to the element.

### `const T& operator[](const std::vector<size_t>& indices) const`
- **Description**: Const version of multi-dimensional index access.
- **Parameters**:
  - `indices`: A vector of indices for each dimension.
- **Returns**: A const reference to the element.

---

## Arithmetic Operators (Element-Wise)

These operators perform element-wise operations and require that both arrays have the same shape. They use multi-threading for efficiency.

### `NDArray<T> operator+(const NDArray<T>& other) const`
- **Description**: Adds two arrays element-wise.
- **Parameters**:
  - `other`: The array to add.
- **Throws**:
  - `std::runtime_error` if shapes do not match.
- **Returns**: A new `NDArray` containing the sum.

### `NDArray<T> operator-(const NDArray<T>& other) const`
- **Description**: Subtracts two arrays element-wise.
- **Parameters**:
  - `other`: The array to subtract.
- **Throws**:
  - `std::runtime_error` if shapes do not match.
- **Returns**: A new `NDArray` containing the difference.

### `NDArray<T> operator*(const NDArray<T>& other) const`
- **Description**: Multiplies two arrays element-wise.
- **Parameters**:
  - `other`: The array to multiply.
- **Throws**:
  - `std::runtime_error` if shapes do not match.
- **Returns**: A new `NDArray` containing the product.

### `NDArray<T> operator/(const NDArray<T>& other) const`
- **Description**: Divides two arrays element-wise.
- **Parameters**:
  - `other`: The array to divide by.
- **Throws**:
  - `std::runtime_error` if shapes do not match.
- **Returns**: A new `NDArray` containing the quotient.

---

## Dot Product (Matrix Multiplication)

### `NDArray<T> dot(const NDArray<T>& other) const`
- **Description**: Computes the matrix multiplication (dot product) of two 2D arrays using multi-threading.
- **Parameters**:
  - `other`: The array to multiply with.
- **Throws**:
  - `std::runtime_error` if the arrays are not 2D or if their shapes do not align for multiplication (i.e., `shape_[1]` must equal `other.shape_[0]`).
- **Returns**: A new `NDArray` containing the result of the matrix multiplication.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr1({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  NumCPP::NDArray<double> arr2({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  NumCPP::NDArray<double> result = arr1.dot(arr2);
  ```

---

## Utility

### `void print() const`
- **Description**: Prints the flattened array to the console.
- **Usage**:
  ```cpp
  NumCPP::NDArray<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
  arr.print(); // Outputs: [1, 2, 3, 4]
  ```

---

## Private Helper Functions

These functions are used internally by the class and are not intended for direct use.

### `std::vector<size_t> compute_strides(const std::vector<size_t>& shape) const`
- **Description**: Computes the strides for the given shape.

### `size_t compute_index(const std::vector<size_t>& indices) const`
- **Description**: Computes the flat index from multi-dimensional indices.

### `T determinant_helper() const`
- **Description**: Helper function to compute the determinant using Gaussian elimination with partial pivoting.

### `void invert_helper()`
- **Description**: Helper function to compute the inverse of the matrix in place using Gaussian elimination with partial pivoting.

---

This documentation provides a comprehensive overview of the `NDArray` class, its methods, and how to use them. For additional details, refer to the source code in `NDArray.tpp` and `NDArray.hpp`.