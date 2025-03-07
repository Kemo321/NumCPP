# NumCPP - C++ Equivalent of NumPy

NumCPP is a C++ library designed to bring the power of numerical computing to C++, drawing inspiration from Python's NumPy. It provides efficient multi-dimensional array handling and a rich set of mathematical operations, all optimized for performance in C++'s compiled environment. As a header-only library, NumCPP integrates seamlessly into C++ projects without requiring additional dependencies or complex build processes.

---

## Overview

NumCPP aims to replicate NumPy's simplicity and functionality while leveraging C++'s strengths, such as type safety, speed, and multi-threading. It is well-suited for applications in scientific computing, data analysis, and machine learning where performance and flexibility are critical.

### Key Features

- **Multi-dimensional Arrays**: Powered by the `NDArray` class for handling arrays of any dimensionality.
- **Mathematical Operations**: Supports element-wise arithmetic, matrix multiplication, and more.
- **Performance**: Optimized with multi-threading and C++'s native capabilities.
- **Ease of Use**: Header-only design for straightforward integration.

---

## The N Dimensional (NDArray) class

The cornerstone of NumCPP is the `NDArray` class, a templated, multi-dimensional array implementation that supports a variety of data types and operations. It provides the foundation for numerical computations, offering methods for:

- Element-wise operations (e.g., addition, subtraction, multiplication)
- Matrix operations (e.g., dot product, transposition)
- Array manipulations (e.g., reshaping, slicing)
- Statistical functions (e.g., sum, mean, min, max)

For detailed documentation on the `NDArray` class, see [NDArray Documentation](docs/NDArray.md) (placeholder link to be expanded in future updates).

### Example Usage

```cpp
#include "NDArray.hpp"

int main() {
    // Create a 2x3 NDArray initialized with zeros
    NumCPP::NDArray<double> arr({2, 3}, 0.0);

    // Create a 2x2 NDArray with specific values
    NumCPP::NDArray<double> arr2({2, 2}, {1.0, 2.0, 3.0, 4.0});

    // Perform matrix multiplication
    NumCPP::NDArray<double> result = arr2.dot(arr2);

    return 0;
}
```

---

## Future Enhancements

NumCPP is an evolving project with plans to expand its functionality by introducing new classes and features inspired by NumPy and tailored to C++'s capabilities.

### Potential New Features


- **Advanced Mathematical Functions**:
  - **Random Number Generation**: For simulations and statistical modeling.
  - **Polynomial Operations**: For fitting and evaluating polynomial equations.
- **Performance Enhancements**:
  - **SIMD Instructions**
  - **Use of CUDA** 

---
