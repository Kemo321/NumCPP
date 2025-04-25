#include <gtest/gtest.h>
#include "Array.hpp"
#include "Matrix.hpp"
#include "SquareMatrix.hpp"
#include <stdexcept>
#include <vector>

namespace NumCPP {

TEST(ArrayTest, Test2DSquare) {
    Array<double> arr({2, 2});
    arr({0, 0}) = 1.0;
    arr({0, 1}) = 2.0;
    arr({1, 0}) = 3.0;
    arr({1, 1}) = 4.0;

    EXPECT_EQ(arr.shape(), std::vector<size_t>({2, 2}));
    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(), 4);

    SquareMatrix<double> sq_mat(arr);
    EXPECT_EQ(sq_mat.determinant(), -2.0);
}

TEST(ArrayTest, TestDeterminantNonSquare) {
    Array<double> arr({2, 3});
    EXPECT_THROW(SquareMatrix<double>{arr}, std::invalid_argument);
}

TEST(ArrayTest, Determinant) {
    Array<double> arr({2, 2});
    arr({0, 0}) = 1.0;
    arr({0, 1}) = 2.0;
    arr({1, 0}) = 3.0;
    arr({1, 1}) = 4.0;
    SquareMatrix<double> sq_mat(arr);
    EXPECT_NEAR(sq_mat.determinant(), -2.0, 0.01);
}

TEST(ArrayTest, TestDeterminant3x3) {
    Array<double> arr({3, 3});
    arr({0, 0}) = 4.0;
    arr({0, 1}) = 3.0;
    arr({0, 2}) = 2.0;
    arr({1, 0}) = 2.0;
    arr({1, 1}) = 5.0;
    arr({1, 2}) = 1.0;
    arr({2, 0}) = 1.0;
    arr({2, 1}) = 2.0;
    arr({2, 2}) = 3.0;
    SquareMatrix<double> sq_mat(arr);
    EXPECT_NEAR(sq_mat.determinant(), 35.0, 0.01);
}

} // namespace NumCPP