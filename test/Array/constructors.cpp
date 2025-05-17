#include "Array.hpp"
#include <gtest/gtest.h>

using namespace NumCPP;

TEST(Constructors, DefaultConstructor)
{
    Array<double> arr;
    EXPECT_TRUE(arr.shape().empty());
    EXPECT_EQ(arr.size(), 0);
}

TEST(Constructors, DefaultConstructorData)
{
    Array<double> arr;
    EXPECT_THROW(arr(0), std::invalid_argument);
}

TEST(Constructors, CopyConstructorTest)
{
    Array<double> arr1({ 2, 3 }, 5.0);
    Array<double> arr2(arr1);
    EXPECT_EQ(arr2.shape(), std::vector<size_t>({ 2, 3 }));
    arr1(0, 0) = 10.0;
    EXPECT_EQ(arr2(0, 0), 5.0); // Deep copy
}

TEST(Constructors, CopyConstructorEmpty)
{
    Array<double> arr1;
    Array<double> arr2(arr1);
    EXPECT_TRUE(arr2.shape().empty());
}

TEST(Constructors, MoveConstructor)
{
    Array<double> arr1({ 2, 3 }, 5.0);
    Array<double> arr2(std::move(arr1));
    EXPECT_EQ(arr2.shape(), std::vector<size_t>({ 2, 3 }));
    EXPECT_EQ(arr2(0, 0), 5.0);
    EXPECT_TRUE(arr1.shape().empty()); // Moved-from state
}

TEST(Constructors, MoveConstructorData)
{
    Array<double> arr1({ 1, 2 }, 3.0);
    Array<double> arr2(std::move(arr1));
    EXPECT_EQ(arr2(0, 1), 3.0);
}

TEST(Constructors, ShapeAndInitVal)
{
    Array<double> arr({ 2, 3 }, 5.0);
    EXPECT_EQ(arr.shape(), std::vector<size_t>({ 2, 3 }));
    EXPECT_EQ(arr(1, 2), 5.0);
}

TEST(Constructors, ShapeAndInitValZero)
{
    Array<double> arr({ 1, 1 }, 0.0);
    EXPECT_EQ(arr(0, 0), 0.0);
}

TEST(Constructors, ShapeAndDataVector)
{
    std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    Array<double> arr({ 2, 3 }, data);
    EXPECT_EQ(arr(0, 0), 1.0);
    EXPECT_EQ(arr(1, 2), 6.0);
}

TEST(Constructors, ShapeAndDataVectorMismatch)
{
    std::vector<double> data = { 1.0, 2.0 };
    EXPECT_THROW(Array<double>({ 2, 2 }, data), std::invalid_argument);
}