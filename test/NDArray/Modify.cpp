#include <gtest/gtest.h>
#include "Array.hpp"
#include "Matrix.hpp"
#include "SquareMatrix.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace NumCPP {

class ArrayModiTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ArrayModiTest, Fill2DArray) {
    Array<double> arr({3, 4}, 0.0);
    arr.fill(2.5);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            EXPECT_DOUBLE_EQ(arr({i, j}), 2.5);
        }
    }
}

TEST_F(ArrayModiTest, Invert1x1Matrix) {
    Array<double> arr({1, 1}, std::vector<double>{5});
    SquareMatrix<double> sq_mat(arr);
    sq_mat.invert();
    EXPECT_NEAR(arr({0, 0}), 0.2, 1e-6);
}

TEST_F(ArrayModiTest, Invert2x2Matrix) {
    Array<double> arr({2, 2}, std::vector<double>{1, 2, 3, 4});
    Array<double> orig = arr.copy();
    SquareMatrix<double> sq_mat(arr);
    sq_mat.invert();
    EXPECT_NEAR(arr({0, 0}), -2, 1e-6);
    EXPECT_NEAR(arr({0, 1}), 1, 1e-6);
    EXPECT_NEAR(arr({1, 0}), 1.5, 1e-6);
    EXPECT_NEAR(arr({1, 1}), -0.5, 1e-6);

    Matrix<double> mat(orig);
    Matrix<double> inv_mat(arr);
    Array<double> identity = mat.dot(inv_mat);
    EXPECT_NEAR(identity({0, 0}), 1, 1e-6);
    EXPECT_NEAR(identity({0, 1}), 0, 1e-6);
    EXPECT_NEAR(identity({1, 0}), 0, 1e-6);
    EXPECT_NEAR(identity({1, 1}), 1, 1e-6);
}

TEST_F(ArrayModiTest, InvertNonSquareMatrix) {
    Array<double> arr({2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    EXPECT_THROW(SquareMatrix<double>{arr}, std::invalid_argument);
}

TEST_F(ArrayModiTest, Invert3x3Matrix) {
    Array<double> arr({3, 3}, std::vector<double>{1, 2, 3, 0, 1, 4, 5, 6, 0});
    Array<double> orig = arr.copy();
    SquareMatrix<double> sq_mat(arr);
    sq_mat.invert();
    Matrix<double> mat(orig);
    Matrix<double> inv_mat(arr);
    Array<double> identity = mat.dot(inv_mat);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (i == j) EXPECT_NEAR(identity({i, j}), 1, 0.01);
            else EXPECT_NEAR(identity({i, j}), 0, 0.01);
        }
    }
}

TEST_F(ArrayModiTest, Inverted2x2Matrix) {
    Array<double> arr({2, 2}, std::vector<double>{1, 2, 3, 4});
    SquareMatrix<double> sq_mat(arr);
    Array<double> inv_arr = sq_mat.inverted();
    Matrix<double> mat(arr);
    Matrix<double> inv_mat(inv_arr);
    Array<double> identity = mat.dot(inv_mat);
    EXPECT_NEAR(identity({0, 0}), 1, 1e-6);
    EXPECT_NEAR(identity({0, 1}), 0, 1e-6);
    EXPECT_NEAR(identity({1, 0}), 0, 1e-6);
    EXPECT_NEAR(identity({1, 1}), 1, 1e-6);
}

TEST_F(ArrayModiTest, DotProduct) {
    Array<double> arr1({2, 3}, std::vector<double>{1, 2, 3, 4, 5, 6});
    Array<double> arr2({3, 2}, std::vector<double>{7, 8, 9, 10, 11, 12});
    Matrix<double> mat1(arr1);
    Matrix<double> mat2(arr2);
    Array<double> result = mat1.dot(mat2);
    EXPECT_EQ(result.shape(), std::vector<size_t>({2, 2}));
    EXPECT_NEAR(result({0, 0}), 58, 1e-6);
    EXPECT_NEAR(result({0, 1}), 64, 1e-6);
    EXPECT_NEAR(result({1, 0}), 139, 1e-6);
    EXPECT_NEAR(result({1, 1}), 154, 1e-6);
}

} // namespace NumCPP