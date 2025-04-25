#include <gtest/gtest.h>
#include "Array.hpp"
#include "Matrix.hpp"
#include "SquareMatrix.hpp"

namespace NumCPP {

TEST(ArrayTest, DefaultConstructor) {
    Array<int> array;
    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.shape().size(), 0);
}

TEST(ArrayTest, ConstructorWithShapeAndInitVal) {
    Array<int> array({3, 3}, 5);
    EXPECT_EQ(array.size(), 9);
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 5);
    }
}

TEST(MatrixTest, ConstructorValid) {
    Array<double> arr({2, 3}, 0.0);
    EXPECT_NO_THROW(Matrix<double>{arr});
}

TEST(MatrixTest, ConstructorInvalid) {
    Array<double> arr({3}, 0.0);
    EXPECT_THROW(Matrix<double>{arr}, std::invalid_argument);
}

TEST(SquareMatrixTest, ConstructorValid) {
    Array<double> arr({2, 2}, 0.0);
    EXPECT_NO_THROW(SquareMatrix<double>{arr});
}

TEST(SquareMatrixTest, ConstructorInvalidNonSquare) {
    Array<double> arr({2, 3}, 0.0);
    EXPECT_THROW(SquareMatrix<double>{arr}, std::invalid_argument);
}

TEST(SquareMatrixTest, ConstructorInvalidNDim) {
    Array<double> arr({2, 2, 2}, 0.0);
    EXPECT_THROW(SquareMatrix<double>{arr}, std::invalid_argument);
}

} // namespace NumCPP