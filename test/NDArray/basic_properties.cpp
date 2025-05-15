#include <gtest/gtest.h>
#include "Array.hpp"

using namespace NumCPP;

TEST(BasicArrayProperties, Shape) {
    Array<double> arr({2, 3});
    EXPECT_EQ(arr.shape(), std::vector<size_t>({2, 3}));
}

TEST(BasicArrayProperties, ShapeEmpty) {
    Array<double> arr;
    EXPECT_TRUE(arr.shape().empty());
}

TEST(BasicArrayProperties, Ndim) {
    Array<double> arr({2, 3});
    EXPECT_EQ(arr.ndim(), 2);
}

TEST(BasicArrayProperties, NdimEmpty) {
    Array<double> arr;
    EXPECT_EQ(arr.ndim(), 0);
}

TEST(BasicArrayProperties, Size) {
    Array<double> arr({2, 3});
    EXPECT_EQ(arr.size(), 6);
}

TEST(BasicArrayProperties, SizeEmpty) {
    Array<double> arr;
    EXPECT_EQ(arr.size(), 0);
}

TEST(BasicArrayProperties, Strides) {
    Array<double> arr({2, 3});
    EXPECT_EQ(arr.strides(), std::vector<size_t>({3, 1}));
}

TEST(BasicArrayProperties, Strides3D) {
    Array<double> arr({2, 3, 4});
    EXPECT_EQ(arr.strides(), std::vector<size_t>({12, 4, 1}));
}

TEST(BasicArrayProperties, Reshape) {
    Array<double> arr({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto reshaped = arr.reshape({3, 2});
    EXPECT_EQ(reshaped.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(reshaped(2, 1), 6.0);
}

TEST(BasicArrayProperties, ReshapeTo1D) {
    Array<double> arr({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto reshaped = arr.reshape({6});
    EXPECT_EQ(reshaped(5), 6.0);
}

TEST(BasicArrayProperties, Flatten) {
    Array<double> arr({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto flat = arr.flatten();
    EXPECT_EQ(flat, std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

TEST(BasicArrayProperties, FlattenEmpty) {
    Array<double> arr;
    auto flat = arr.flatten();
    EXPECT_TRUE(flat.empty());
}

TEST(BasicArrayProperties, Sum) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr.sum(), 10.0);
}

TEST(BasicArrayProperties, SumNegative) {
    Array<double> arr({2}, {-1.0, -2.0});
    EXPECT_EQ(arr.sum(), -3.0);
}

TEST(BasicArrayProperties, Mean) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr.mean(), 2.5);
}

TEST(BasicArrayProperties, MeanEmpty) {
    Array<double> arr;
    EXPECT_THROW(arr.mean(), std::runtime_error);
}

TEST(BasicArrayProperties, Min) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr.min(), 1.0);
}

TEST(BasicArrayProperties, MinNegative) {
    Array<double> arr({2}, {-1.0, -2.0});
    EXPECT_EQ(arr.min(), -2.0);
}

TEST(BasicArrayProperties, Max) {
    Array<double> arr({2, 2}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr.max(), 4.0);
}

TEST(BasicArrayProperties, MaxEmpty) {
    Array<double> arr;
    EXPECT_THROW(arr.max(), std::runtime_error);
}

TEST(BasicArrayProperties, IsSquare) {
    Array<double> arr({2, 2});
    EXPECT_TRUE(arr.is_square());
}

TEST(BasicArrayProperties, IsSquareFalse) {
    Array<double> arr({2, 3});
    EXPECT_FALSE(arr.is_square());
}