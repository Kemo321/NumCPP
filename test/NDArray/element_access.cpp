#include <gtest/gtest.h>
#include "Array.hpp"

using namespace NumCPP;

TEST(ElementAccess, OneDimensionalParentheses) {
    Array<double> arr({3}, {1.0, 2.0, 3.0});
    EXPECT_EQ(arr(1), 2.0);
    arr(1) = 5.0;
    EXPECT_EQ(arr(1), 5.0);
}

TEST(ElementAccess, OneDimensionalParenthesesOutOfRange) {
    Array<double> arr({2}, {1.0, 2.0});
    EXPECT_THROW(arr(2), std::runtime_error);
}

TEST(ElementAccess, MultiDimensionalParentheses) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr({1, 1}), 4.0);
    arr({0, 1}) = 5.0;
    EXPECT_EQ(arr({0, 1}), 5.0);
}

TEST(ElementAccess, MultiDimensionalParenthesesInvalidIndices) {
    Array<double> arr({2, 2});
    EXPECT_THROW(arr({2, 0}), std::runtime_error);
}

TEST(ElementAccess, OneDimensionalBrackets) {
    Array<double> arr({3}, {1.0, 2.0, 3.0});
    EXPECT_EQ(arr[1], 2.0);
    arr[1] = 5.0;
    EXPECT_EQ(arr[1], 5.0);
}

TEST(ElementAccess, OneDimensionalBracketsOutOfRange) {
    Array<double> arr({2}, {1.0, 2.0});
    EXPECT_THROW(arr[2], std::out_of_range);
}

TEST(ElementAccess, MultiDimensionalBrackets) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr(1, 1), 4.0);
    arr[{0, 1}] = 5.0;
    EXPECT_EQ(arr(0, 1), 5.0);
}

TEST(ElementAccess, MultiDimensionalBracketsWrongDim) {
    Array<double> arr({2, 2});
    EXPECT_NO_THROW(arr[0]);
}