#include "Array.hpp"
#include <gtest/gtest.h>

using namespace NumCPP;

TEST(ModifyArray, Fill)
{
    Array<double> arr({ 2, 2 });
    arr.fill(10.0);
    EXPECT_EQ(arr(1, 1), 10.0);
}

TEST(ModifyArray, FillEmpty)
{
    Array<double> arr;
    arr.fill(5.0); // Should do nothing
    EXPECT_EQ(arr.size(), 0);
}

TEST(ModifyArray, Zeros)
{
    Array<double> arr({ 2, 2 }, 5.0);
    arr.zeros();
    EXPECT_EQ(arr(1, 1), 0.0);
}

TEST(ModifyArray, ZerosAlreadyZero)
{
    Array<double> arr({ 1, 1 }, 0.0);
    arr.zeros();
    EXPECT_EQ(arr(0, 0), 0.0);
}

TEST(ModifyArray, Ones)
{
    Array<double> arr({ 2, 2 }, 5.0);
    arr.ones();
    EXPECT_EQ(arr(1, 1), 1.0);
}

TEST(ModifyArray, OnesEmpty)
{
    Array<double> arr;
    arr.ones();
    EXPECT_EQ(arr.size(), 0);
}

TEST(ModifyArray, Transpose)
{
    Array<double> arr({ 2, 3 }, { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    arr.transpose();
    EXPECT_EQ(arr.shape(), std::vector<size_t>({ 3, 2 }));
    EXPECT_EQ(arr(2, 1), 6.0);
}

TEST(ModifyArray, TransposeSquare)
{
    Array<double> arr({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    arr.transpose();
    EXPECT_EQ(arr(0, 1), 3.0);
}

TEST(ModifyArray, Reverse)
{
    Array<double> arr({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    arr.reverse();
    EXPECT_EQ(arr(0, 0), 4.0);
}

TEST(ModifyArray, ReverseEmpty)
{
    Array<double> arr;
    arr.reverse();
    EXPECT_TRUE(arr.shape().empty());
}

TEST(ModifyArray, Pow)
{
    Array<double> arr({ 2, 2 }, { 2.0, 3.0, 4.0, 5.0 });
    arr.pow(2.0);
    EXPECT_EQ(arr(1, 1), 25.0);
}

TEST(ModifyArray, PowZero)
{
    Array<double> arr({ 1, 1 }, 5.0);
    arr.pow(0.0);
    EXPECT_EQ(arr(0, 0), 1.0);
}