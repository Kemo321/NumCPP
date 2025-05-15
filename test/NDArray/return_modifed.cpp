#include <gtest/gtest.h>
#include "Array.hpp"

using namespace NumCPP;

TEST(ModifyArray, mFill) {
    Array<double> arr({2, 2});
    arr.fill(10.0);
    EXPECT_EQ(arr(1, 1), 10.0);
}

TEST(ModifyArray, mFillEmpty) {
    Array<double> arr;
    arr.fill(5.0);  // Should do nothing
    EXPECT_EQ(arr.size(), 0);
}

TEST(ModifyArray, mZeros) {
    Array<double> arr({2, 2}, 5.0);
    arr.zeros();
    EXPECT_EQ(arr(1, 1), 0.0);
}

TEST(ModifyArray, mZerosAlreadyZero) {
    Array<double> arr({1, 1}, 0.0);
    arr.zeros();
    EXPECT_EQ(arr(0, 0), 0.0);
}

TEST(ModifyArray, mOnes) {
    Array<double> arr({2, 2}, 5.0);
    arr.ones();
    EXPECT_EQ(arr(1, 1), 1.0);
}

TEST(ModifyArray, mOnesEmpty) {
    Array<double> arr;
    arr.ones();
    EXPECT_EQ(arr.size(), 0);
}

TEST(ModifyArray, mTranspose) {
    Array<double> arr({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    arr.transpose();
    EXPECT_EQ(arr.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(arr(2, 1), 6.0);
}

TEST(ModifyArray, mTransposeSquare) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    arr.transpose();
    EXPECT_EQ(arr(0, 1), 3.0);
}

TEST(ModifyArray, mReverse) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    arr.reverse();
    EXPECT_EQ(arr(0, 0), 4.0);
}

TEST(ModifyArray, mReverseEmpty) {
    Array<double> arr;
    arr.reverse();
    EXPECT_TRUE(arr.shape().empty());
}

TEST(ModifyArray, mPow) {
    Array<double> arr({2, 2}, {2.0, 3.0, 4.0, 5.0});
    arr.pow(2.0);
    EXPECT_EQ(arr(1, 1), 25.0);
}

TEST(ModifyArray, mPowZero) {
    Array<double> arr({1, 1}, 5.0);
    arr.pow(0.0);
    EXPECT_EQ(arr(0, 0), 1.0);
}