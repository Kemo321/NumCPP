#include <gtest/gtest.h>
#include "NDArray.hpp"
#include <stdexcept>
#include <vector>

// Test the shape, ndim, and size for 1D array
TEST(NDArrayTest, Test1DArray) {
    NumCPP::NDArray<double> arr({5});
    arr(0) = 1.0;
    arr(1) = 2.0;
    arr(2) = 3.0;
    arr(3) = 4.0;
    arr(4) = 5.0;

    // Shape, size, and ndim
    EXPECT_EQ(arr.shape(), std::vector<size_t>({5}));
    EXPECT_EQ(arr.ndim(), 1);
    EXPECT_EQ(arr.size(), 5);

    // Sum, mean, min, max
    EXPECT_EQ(arr.sum(), 15.0);
    EXPECT_EQ(arr.mean(), 3.0);
    EXPECT_EQ(arr.min(), 1.0);
    EXPECT_EQ(arr.max(), 5.0);

    // Flatten
    auto flat1D = arr.flatten();
    EXPECT_EQ(flat1D, std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0}));

    // Reshape
    auto reshaped = arr.reshape({1, 5});
    EXPECT_EQ(reshaped.shape(), std::vector<size_t>({1, 5}));
    EXPECT_EQ(reshaped({0, 0}), 1.0);
    EXPECT_EQ(reshaped({0, 1}), 2.0);
    EXPECT_EQ(reshaped({0, 2}), 3.0);
    EXPECT_EQ(reshaped({0, 3}), 4.0);
    EXPECT_EQ(reshaped({0, 4}), 5.0);

    // Test reshape with invalid shape
    EXPECT_THROW(arr.reshape({2, 3}), std::runtime_error);

    // Test determinant for 1D array (throws exception)
    EXPECT_THROW(arr.determinant(), std::runtime_error);

    // Test sum, mean, min, max with multi-threading
}

// Test the shape, ndim, and size for 2D array
TEST(NDArrayTest, Test2DArray) {
    NumCPP::NDArray<double> arr({2, 3});
    arr({0, 0}) = 1.0;
    arr({0, 1}) = 2.0;
    arr({0, 2}) = 3.0;
    arr({1, 0}) = 4.0;
    arr({1, 1}) = 5.0;
    arr({1, 2}) = 6.0;

    // Shape, size, and ndim
    EXPECT_EQ(arr.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(), 6);

    // Sum, mean, min, max
    EXPECT_EQ(arr.sum(), 21.0);
    EXPECT_EQ(arr.mean(), 3.5);
    EXPECT_EQ(arr.min(), 1.0);
    EXPECT_EQ(arr.max(), 6.0);

    // Flatten
    auto flat2D = arr.flatten();
    EXPECT_EQ(flat2D, std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));

    // The determinant is not defined for non-square matrices
    EXPECT_THROW(arr.determinant(), std::runtime_error);

    // Reshape
    auto reshaped = arr.reshape({3, 2});
    EXPECT_EQ(reshaped.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(reshaped({0, 0}), 1.0);
    EXPECT_EQ(reshaped({0, 1}), 2.0);
    EXPECT_EQ(reshaped({1, 0}), 3.0);
    EXPECT_EQ(reshaped({1, 1}), 4.0);
    EXPECT_EQ(reshaped({2, 0}), 5.0);
    EXPECT_EQ(reshaped({2, 1}), 6.0);
}


TEST(NDArrayTest, Test2DSquare){
    NumCPP::NDArray<double> arr({2, 2});
    arr({0, 0}) = 1.0;
    arr({0, 1}) = 2.0;
    arr({1, 0}) = 3.0;
    arr({1, 1}) = 4.0;

    // Shape, size, and ndim
    EXPECT_EQ(arr.shape(), std::vector<size_t>({2, 2}));
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(), 4);

    // Sum, mean, min, max
    EXPECT_EQ(arr.sum(), 10.0);
    EXPECT_EQ(arr.mean(), 2.5);
    EXPECT_EQ(arr.min(), 1.0);
    EXPECT_EQ(arr.max(), 4.0);

    // Flatten
    auto flat2D = arr.flatten();
    EXPECT_EQ(flat2D, std::vector<double>({1.0, 2.0, 3.0, 4.0}));

    // The determinant is not defined for non-square matrices
    EXPECT_EQ(arr.determinant(), -2.0);

    // Reshape
    auto reshaped = arr.reshape({1, 4});
    EXPECT_EQ(reshaped.shape(), std::vector<size_t>({1, 4}));
    EXPECT_EQ(reshaped({0, 0}), 1.0);
    EXPECT_EQ(reshaped({0, 1}), 2.0);
    EXPECT_EQ(reshaped({0, 2}), 3.0);
    EXPECT_EQ(reshaped({0, 3}), 4.0);    
}


// Test determinant for non-square matrix (throws exception)
TEST(NDArrayTest, TestDeterminantNonSquare) {
    NumCPP::NDArray<double> arr({2, 3});
    EXPECT_THROW(arr.determinant(), std::runtime_error);
}

// Test sum, mean, min, max with multi-threading
TEST(NDArrayTest, TestMultiThreadingSum) {
    NumCPP::NDArray<double> arr({5});
    arr(0) = 1.0;
    arr(1) = 2.0;
    arr(2) = 3.0;
    arr(3) = 4.0;
    arr(4) = 5.0;

    EXPECT_EQ(arr.sum(), 15.0);
    EXPECT_EQ(arr.mean(), 3.0);
    EXPECT_EQ(arr.min(), 1.0);
    EXPECT_EQ(arr.max(), 5.0);
}

// Test for invalid reshaping (throw exception)
TEST(NDArrayTest, TestReshapeIncompatible) {
    NumCPP::NDArray<double> arr({6});
    arr(0) = 1.0;
    arr(1) = 2.0;
    arr(2) = 3.0;
    arr(3) = 4.0;
    arr(4) = 5.0;
    arr(5) = 6.0;

    EXPECT_THROW(arr.reshape({2, 4}), std::runtime_error);
}

// Test is_square method
TEST(NDArrayTest, TestIsSquare) {
    NumCPP::NDArray<double> arr({2, 2});
    EXPECT_TRUE(arr.is_square());

    NumCPP::NDArray<double> arr2({2, 3});
    EXPECT_FALSE(arr2.is_square());

    NumCPP::NDArray<double> arr3({3, 2, 3, 2});
    EXPECT_FALSE(arr3.is_square());

    NumCPP::NDArray<double> arr4({3, 3, 3, 3, 3});
    EXPECT_TRUE(arr4.is_square());
}


TEST(NDArrayTest, determinant)
{
    NumCPP::NDArray<double> arr({2, 2});
    arr({0, 0}) = 1.0;
    arr({0, 1}) = 2.0;
    arr({1, 0}) = 3.0;
    arr({1, 1}) = 4.0;

    EXPECT_NEAR(arr.determinant(), -2.0, 0.01);

    NumCPP::NDArray<double> arr2({3, 3});
    arr2({0, 0}) = 1.0;
    arr2({0, 1}) = 2.0;
    arr2({0, 2}) = 3.0;
    arr2({1, 0}) = 4.0;
    arr2({1, 1}) = 5.0;
    arr2({1, 2}) = 6.0;
    arr2({2, 0}) = 7.0;
    arr2({2, 1}) = 8.0;
    arr2({2, 2}) = 9.0;

    EXPECT_NEAR(arr2.determinant(), 0.0, 0.01);

    NumCPP::NDArray<double> arr3({4, 4});
    arr3({0, 0}) = 1.0;
    arr3({0, 1}) = 2.0;
    arr3({0, 2}) = 3.0;
    arr3({0, 3}) = 4.0;
    arr3({1, 0}) = 5.0;
    arr3({1, 1}) = 6.0;
    arr3({1, 2}) = 7.0;
    arr3({1, 3}) = 8.0;
    arr3({2, 0}) = 9.0;
    arr3({2, 1}) = 10.0;
    arr3({2, 2}) = 11.0;
    arr3({2, 3}) = 12.0;
    arr3({3, 0}) = 13.0;
    arr3({3, 1}) = 14.0;
    arr3({3, 2}) = 15.0;
    arr3({3, 3}) = 16.0;

    EXPECT_NEAR(arr3.determinant(), 0.0, 0.01);
}

// Test determinant for a 3x3 matrix
TEST(NDArrayTest, TestDeterminant3x3) {
    NumCPP::NDArray<double> arr({3, 3});
    arr({0, 0}) = 4.0;
    arr({0, 1}) = 3.0;
    arr({0, 2}) = 2.0;
    arr({1, 0}) = 2.0;
    arr({1, 1}) = 5.0;
    arr({1, 2}) = 1.0;
    arr({2, 0}) = 1.0;
    arr({2, 1}) = 2.0;
    arr({2, 2}) = 3.0;

    EXPECT_NEAR(arr.determinant(), 35.0, 0.01);
}

// Test determinant for a 4x4 matrix
TEST(NDArrayTest, TestDeterminant4x4) {
    NumCPP::NDArray<double> arr({4, 4});
    arr({0, 0}) = 6.0;
    arr({0, 1}) = 1.0;
    arr({0, 2}) = 3.0;
    arr({0, 3}) = 4.0;
    arr({1, 0}) = 2.0;
    arr({1, 1}) = 1.0;
    arr({1, 2}) = 1.0;
    arr({1, 3}) = 3.0;
    arr({2, 0}) = 3.0;
    arr({2, 1}) = 2.0;
    arr({2, 2}) = 4.0;
    arr({2, 3}) = 2.0;
    arr({3, 0}) = 5.0;
    arr({3, 1}) = 3.0;
    arr({3, 2}) = 7.0;
    arr({3, 3}) = 1.0;

    EXPECT_NEAR(arr.determinant(), -10, 0.01);
}

// Test determinant for another 3x3 matrix
TEST(NDArrayTest, TestDeterminantAnother3x3) {
    NumCPP::NDArray<double> arr({3, 3});
    arr({0, 0}) = 1.0;
    arr({0, 1}) = 2.0;
    arr({0, 2}) = 3.0;
    arr({1, 0}) = 1.0;
    arr({1, 1}) = 4.0;
    arr({1, 2}) = 2.0;
    arr({2, 0}) = 3.0;
    arr({2, 1}) = 1.0;
    arr({2, 2}) = 5.0;

    EXPECT_NEAR(arr.determinant(), -13.0, 0.01);
}

// Test determinant for a specific 4x4 matrix
TEST(NDArrayTest, TestDeterminantSpecific4x4) {
    NumCPP::NDArray<double> arr({4, 4});
    arr({0, 0}) = 1.0;
    arr({0, 1}) = -10.0;
    arr({0, 2}) = 4.0;
    arr({0, 3}) = 23.0;
    arr({1, 0}) = 4.0;
    arr({1, 1}) = -124.0;
    arr({1, 2}) = 5.0;
    arr({1, 3}) = 24.0;
    arr({2, 0}) = 1234.0;
    arr({2, 1}) = 423.0;
    arr({2, 2}) = -42.0;
    arr({2, 3}) = 12.0;
    arr({3, 0}) = 1.0;
    arr({3, 1}) = 2.0;
    arr({3, 2}) = -2.0;
    arr({3, 3}) = -42.0;

    EXPECT_NEAR(arr.determinant(), -17116849, 1);
}