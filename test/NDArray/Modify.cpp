#include <gtest/gtest.h>
#include "NDArray.tpp" // Adjust path as needed (e.g., "../include/NDArray.tpp")
#include <vector>
#include <cmath>
#include <stdexcept>

namespace NumCPP {

// Helper function to multiply 2x2 matrices for invert testing
template<typename T>
NDArray<T> matrix_multiply_2x2(const NDArray<T>& a, const NDArray<T>& b) {
    NDArray<T> result({2, 2}, T(0));
    result({0, 0}) = a({0, 0}) * b({0, 0}) + a({0, 1}) * b({1, 0});
    result({0, 1}) = a({0, 0}) * b({0, 1}) + a({0, 1}) * b({1, 1});
    result({1, 0}) = a({1, 0}) * b({0, 0}) + a({1, 1}) * b({1, 0});
    result({1, 1}) = a({1, 0}) * b({0, 1}) + a({1, 1}) * b({1, 1});
    return result;
}

} // namespace NumCPP

// Test fixture for NDArray tests
class NDArrayModiTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test default constructor
TEST_F(NDArrayModiTest, DefaultConstructor) {
    NumCPP::NDArray<double> arr;
    EXPECT_EQ(arr.size(), 0u);
    EXPECT_EQ(arr.shape(), std::vector<size_t>{});
    EXPECT_EQ(arr.ndim(), 0u);
}

// Test fill method
TEST_F(NDArrayModiTest, Fill1DArray) {
    NumCPP::NDArray<double> arr({10}, 0.0);
    arr.fill(5.0);
    for (size_t i = 0; i < 10; i++) {
        EXPECT_DOUBLE_EQ(arr(i), 5.0);
    }
}

TEST_F(NDArrayModiTest, Fill2DArray) {
    NumCPP::NDArray<double> arr({3, 4}, 0.0);
    arr.fill(2.5);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            EXPECT_DOUBLE_EQ(arr({i, j}), 2.5);
        }
    }
}

TEST_F(NDArrayModiTest, FillEmptyArray) {
    NumCPP::NDArray<double> arr({}, 0.0);
    arr.fill(1.0);
    EXPECT_EQ(arr.size(), 0u);
}

TEST_F(NDArrayModiTest, FillSingleElement) {
    NumCPP::NDArray<double> arr({1}, 0.0);
    arr.fill(3.14);
    EXPECT_DOUBLE_EQ(arr(0), 3.14);
}

// Test zeros method
TEST_F(NDArrayModiTest, Zeros1DArray) {
    NumCPP::NDArray<int> arr({5}, 1);
    arr.zeros();
    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(arr(i), 0);
    }
}

TEST_F(NDArrayModiTest, Zeros2DArray) {
    NumCPP::NDArray<double> arr({2, 3}, 1.0);
    arr.zeros();
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(arr({i, j}), 0.0);
        }
    }
}

// Test ones method
TEST_F(NDArrayModiTest, Ones1DArray) {
    NumCPP::NDArray<int> arr({5}, 0);
    arr.ones();
    for (size_t i = 0; i < 5; i++) {
        EXPECT_EQ(arr(i), 1);
    }
}

TEST_F(NDArrayModiTest, Ones2DArray) {
    NumCPP::NDArray<double> arr({2, 3}, 0.0);
    arr.ones();
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_DOUBLE_EQ(arr({i, j}), 1.0);
        }
    }
}

// Test transpose method
TEST_F(NDArrayModiTest, Transpose2DArray) {
    NumCPP::NDArray<double> arr({2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    arr.transpose();
    EXPECT_EQ(arr.shape(), std::vector<size_t>({3, 2}));
    EXPECT_DOUBLE_EQ(arr({0, 0}), 1);
    EXPECT_DOUBLE_EQ(arr({0, 1}), 4);
    EXPECT_DOUBLE_EQ(arr({1, 0}), 2);
    EXPECT_DOUBLE_EQ(arr({1, 1}), 5);
    EXPECT_DOUBLE_EQ(arr({2, 0}), 3);
    EXPECT_DOUBLE_EQ(arr({2, 1}), 6);
}

TEST_F(NDArrayModiTest, Transpose1DArray) {
    NumCPP::NDArray<double> arr({4}, std::vector<double>{1, 2, 3, 4});
    arr.transpose();
    EXPECT_EQ(arr.shape(), std::vector<size_t>({4}));
    EXPECT_DOUBLE_EQ(arr(0), 1);
    EXPECT_DOUBLE_EQ(arr(1), 2);
    EXPECT_DOUBLE_EQ(arr(2), 3);
    EXPECT_DOUBLE_EQ(arr(3), 4);
}

// Test reverse method
TEST_F(NDArrayModiTest, Reverse1DArray) {
    NumCPP::NDArray<double> arr({4}, std::vector<double>{1, 2, 3, 4});
    arr.reverse();
    EXPECT_DOUBLE_EQ(arr(0), 4);
    EXPECT_DOUBLE_EQ(arr(1), 3);
    EXPECT_DOUBLE_EQ(arr(2), 2);
    EXPECT_DOUBLE_EQ(arr(3), 1);
}

TEST_F(NDArrayModiTest, Reverse2DArray) {
    NumCPP::NDArray<double> arr({2, 2}, std::vector<double>{1, 2, 3, 4});
    arr.reverse();
    EXPECT_DOUBLE_EQ(arr({0, 0}), 4);
    EXPECT_DOUBLE_EQ(arr({0, 1}), 3);
    EXPECT_DOUBLE_EQ(arr({1, 0}), 2);
    EXPECT_DOUBLE_EQ(arr({1, 1}), 1);
}

TEST_F(NDArrayModiTest, ReverseOddSize) {
    NumCPP::NDArray<double> arr({5}, std::vector<double>{1, 2, 3, 4, 5});
    arr.reverse();
    EXPECT_DOUBLE_EQ(arr(0), 5);
    EXPECT_DOUBLE_EQ(arr(1), 4);
    EXPECT_DOUBLE_EQ(arr(2), 3);
    EXPECT_DOUBLE_EQ(arr(3), 2);
    EXPECT_DOUBLE_EQ(arr(4), 1);
}

// Test pow method
TEST_F(NDArrayModiTest, PowExponent2) {
    NumCPP::NDArray<double> arr({3}, std::vector<double>{1, 2, 3});
    arr.pow(2);
    EXPECT_NEAR(arr(0), 1, 1e-6);
    EXPECT_NEAR(arr(1), 4, 1e-6);
    EXPECT_NEAR(arr(2), 9, 1e-6);
}

TEST_F(NDArrayModiTest, PowExponent0) {
    NumCPP::NDArray<double> arr({3}, std::vector<double>{1, 2, 3});
    arr.pow(0);
    EXPECT_NEAR(arr(0), 1, 1e-6);
    EXPECT_NEAR(arr(1), 1, 1e-6);
    EXPECT_NEAR(arr(2), 1, 1e-6);
}

TEST_F(NDArrayModiTest, PowExponentNegative1) {
    NumCPP::NDArray<double> arr({3}, std::vector<double>{1, 2, 4});
    arr.pow(-1);
    EXPECT_NEAR(arr(0), 1, 1e-6);
    EXPECT_NEAR(arr(1), 0.5, 1e-6);
    EXPECT_NEAR(arr(2), 0.25, 1e-6);
}

// Test invert method
TEST_F(NDArrayModiTest, Invert1x1Matrix) {
    NumCPP::NDArray<double> arr({1, 1}, std::vector<double>{5});
    arr.invert();
    EXPECT_NEAR(arr({0, 0}), 0.2, 1e-6);
}

TEST_F(NDArrayModiTest, Invert2x2Matrix) {
    NumCPP::NDArray<double> arr({2, 2}, std::vector<double>{1, 2, 3, 4});
    NumCPP::NDArray<double> orig = arr;
    arr.invert();
    EXPECT_NEAR(arr({0, 0}), -2, 1e-6);
    EXPECT_NEAR(arr({0, 1}), 1, 1e-6);
    EXPECT_NEAR(arr({1, 0}), 1.5, 1e-6);
    EXPECT_NEAR(arr({1, 1}), -0.5, 1e-6);

    NumCPP::NDArray<double> identity = NumCPP::matrix_multiply_2x2(orig, arr);
    EXPECT_NEAR(identity({0, 0}), 1, 1e-6);
    EXPECT_NEAR(identity({0, 1}), 0, 1e-6);
    EXPECT_NEAR(identity({1, 0}), 0, 1e-6);
    EXPECT_NEAR(identity({1, 1}), 1, 1e-6);
}

TEST_F(NDArrayModiTest, InvertSingularMatrix) {
    NumCPP::NDArray<double> arr({2, 2}, std::vector<double>{1, 2, 2, 4});
    EXPECT_THROW(arr.invert(), std::runtime_error);
}

TEST_F(NDArrayModiTest, InvertNonSquareMatrix) {
    NumCPP::NDArray<double> arr({2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    EXPECT_THROW(arr.invert(), std::invalid_argument);
}

TEST_F(NDArrayModiTest, Invert3x3Matrix) {
    NumCPP::NDArray<double> arr({3, 3}, std::vector<double>{1, 2, 3, 0, 1, 4, 5, 6, 0});
    NumCPP::NDArray<double> orig = arr;
    arr.invert();
    NumCPP::NDArray<double> identity = arr.dot(orig);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (i == j) {
                EXPECT_NEAR(identity({i, j}), 1, 0.01);
            } else {
                EXPECT_NEAR(identity({i, j}), 0, 0.01);
            }
        }
    }
}

// Test multithreading robustness
TEST_F(NDArrayModiTest, FillLargeArrayMultithreaded) {
    NumCPP::NDArray<double> arr({10000}, 0.0);
    arr.fill(1.0);
    for (size_t i = 0; i < 10000; i++) {
        EXPECT_DOUBLE_EQ(arr(i), 1.0);
    }
}

TEST_F(NDArrayModiTest, PowLargeArrayMultithreaded) {
    NumCPP::NDArray<double> arr({10000}, 2.0);
    arr.pow(3);
    for (size_t i = 0; i < 10000; i++) {
        EXPECT_NEAR(arr(i), 8.0, 1e-6);
    }
}
