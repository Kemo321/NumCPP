# NumCPP

NumCPP is a lightweight, header-only C++23 library for numerical computations, providing N-dimensional array operations, matrix manipulations, and specialized square matrix functionality. Inspired by NumPy, it aims to offer efficient and flexible array-based computations in modern C++.

## Features

- **N-Dimensional Arrays**: Create and manipulate arrays of arbitrary dimensions with the `Array` class.
- **Matrix Operations**: Perform 2D matrix operations (e.g., dot product) using the `Matrix` class.
- **Square Matrix Operations**: Compute determinants and inverses with the `SquareMatrix` class.
- **Threaded Computations**: Leverage multi-threading for performance in operations like sum, min, max, and element-wise arithmetic.
- **C++23 Compatibility**: Uses modern C++23 features for clean, efficient code.
- **Header-Only**: No external dependencies except for testing (Google Test).

## Requirements

- C++23-compliant compiler (e.g., Clang 16+, GCC 13+, MSVC 2022+)
- CMake 3.20 or higher
- Google Test 1.11.0 (for running tests)
- Optional: Git for cloning the repository

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/NumCPP.git
   cd NumCPP
   ```

2. **Build and Run Tests**:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ctest
   ```

3. **Use in Your Project**:
   - Copy the `include/` directory to your project.
   - Include the headers (e.g., `#include "Array.hpp"`) and ensure C++23 is enabled.
   - Alternatively, add NumCPP as a subdirectory in your CMake project:
     ```cmake
     add_subdirectory(NumCPP)
     target_include_directories(your_target PRIVATE NumCPP/include)
     ```

## Usage Example

```cpp
#include <Array.hpp>
#include <Matrix.hpp>
#include <SquareMatrix.hpp>
#include <vector>
#include <iostream>

int main() {
    // Create a 2x3 array
    NumCPP::Array<double> arr({2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    arr.print(); // [1, 2, 3, 4, 5, 6]

    // Perform element-wise addition
    NumCPP::Array<double> arr2 = arr + arr;
    arr2.print(); // [2, 4, 6, 8, 10, 12]

    // Create a 2x2 square matrix
    NumCPP::Array<double> sq_arr({2, 2}, std::vector<double>{1, 2, 3, 4});
    NumCPP::SquareMatrix<double> sq_mat(sq_arr);
    std::cout << "Determinant: " << sq_mat.determinant() << '\n'; // Determinant: -2

    // Matrix dot product
    NumCPP::Matrix<double> mat(sq_arr);
    NumCPP::Array<double> result = mat.dot(mat);
    result.print(); // [7, 10, 15, 22]

    return 0;
}
```

## Project Structure

```
NumCPP/
├── include/
│   ├── Array.hpp
│   ├── Array.tpp
│   ├── Matrix.hpp
│   ├── Matrix.tpp
│   ├── SquareMatrix.hpp
│   ├── SquareMatrix.tpp
├── test/
│   ├── NDArray/
│   │   ├── ConDes.cpp
│   │   ├── Modify.cpp
│   │   ├── Access.cpp
│   │   ├── Prop.cpp
├── CMakeLists.txt
├── README.md
```

## Testing

NumCPP includes a test suite using Google Test 1.11.0, located in `test/NDArray/`. Tests cover:
- Array construction and destruction (`ConDes.cpp`)
- Array modifications (e.g., fill, transpose) (`Modify.cpp`)
- Element access and indexing (`Access.cpp`)
- Matrix and square matrix properties (e.g., determinant) (`Prop.cpp`)

Run tests with:
```bash
cd build
ctest
```
