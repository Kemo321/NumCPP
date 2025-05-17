#include "Array.hpp"
#include <gtest/gtest.h>

using namespace NumCPP;

TEST(ArithmeticOperators, ElementWiseAddition)
{
    Array<double> arr1({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    Array<double> arr2({ 2, 2 }, { 5.0, 6.0, 7.0, 8.0 });
    auto result = arr1 + arr2;
    EXPECT_EQ(result(1, 1), 12.0);
}

TEST(ArithmeticOperators, ElementWiseAdditionNegative)
{
    Array<double> arr1({ 2 }, { 1.0, 2.0 });
    Array<double> arr2({ 2 }, { -3.0, -4.0 });
    auto result = arr1 + arr2;
    EXPECT_EQ(result(1), -2.0);
}

TEST(ArithmeticOperators, ElementWiseAdditionScalar)
{
    Array<double> arr({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    auto result = arr + 10.0;
    EXPECT_EQ(result(1, 1), 14.0);
}

TEST(ArithmeticOperators, ElementWiseAdditionScalarNegative)
{
    Array<double> arr({ 1, 1 }, { 1.0 });
    auto result = arr + (-2.0);
    EXPECT_EQ(result(0, 0), -1.0);
}

TEST(ArithmeticOperators, ElementWiseAdditionEmpty)
{
    Array<double> arr1;
    Array<double> arr2;
    auto result = arr1 + arr2;
    EXPECT_TRUE(result.shape().empty());
}

TEST(ArithmeticOperators, ElementWiseAdditionEmptyNonEmpty)
{
    Array<double> arr1({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    double scalar = 0.0;
    auto result = arr1 + scalar;
    EXPECT_EQ(result(1, 1), 4.0);
}

TEST(ArithmeticOperators, ElementWiseAdditionEmptyNonEmptyScalar)
{
    Array<double> arr1({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    double scalar = 10.0;
    auto result = arr1 + scalar;
    EXPECT_EQ(result(1, 1), 14.0);
}

TEST(ArithmeticOperators, ElementWiseAdditionEmptyNonEmptyScalarNegative)
{
    Array<double> arr1({ 2 }, { 1.0, 2.0 });
    double scalar = -3.0;
    auto result = arr1 + scalar;
    EXPECT_EQ(result(1), -1.0);
}

TEST(ArithmeticOperators, ElementWiseAdditionMismatch)
{
    Array<double> arr1({ 2, 2 });
    Array<double> arr2({ 2, 3 });
    EXPECT_THROW(arr1 + arr2, std::runtime_error);
}

TEST(ArithmeticOperators, ElementWiseSubtraction)
{
    Array<double> arr1({ 2, 2 }, { 5.0, 6.0, 7.0, 8.0 });
    Array<double> arr2({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    auto result = arr1 - arr2;
    EXPECT_EQ(result(1, 1), 4.0);
}

TEST(ArithmeticOperators, ElementWiseSubtractionNegative)
{
    Array<double> arr1({ 2 }, { 1.0, 2.0 });
    Array<double> arr2({ 2 }, { 3.0, 4.0 });
    auto result = arr1 - arr2;
    EXPECT_EQ(result(1), -2.0);
}

TEST(ArithmeticOperators, ScalarAddition)
{
    Array<double> arr({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    auto result = arr + 10.0;
    EXPECT_EQ(result(1, 1), 14.0);
}

TEST(ArithmeticOperators, ScalarAdditionNegative)
{
    Array<double> arr({ 1, 1 }, { 1.0 });
    auto result = arr + (-2.0);
    EXPECT_EQ(result(0, 0), -1.0);
}

TEST(ArithmeticOperators, UnaryNegation)
{
    Array<double> arr({ 2, 2 }, { 1.0, -2.0, 3.0, -4.0 });
    auto result = -arr;
    EXPECT_EQ(result(1, 1), 4.0);
}

TEST(ArithmeticOperators, UnaryNegationZero)
{
    Array<double> arr({ 1, 1 }, { 0.0 });
    auto result = -arr;
    EXPECT_EQ(result(0, 0), 0.0);
}

TEST(ArithmeticOperators, PreIncrement)
{
    Array<double> arr({ 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    auto result = ++arr;
    EXPECT_EQ(result(1, 1), 5.0);
    EXPECT_EQ(arr(1, 1), 5.0); // Original modified
}

TEST(ArithmeticOperators, PreIncrementEmpty)
{
    Array<double> arr;
    auto result = ++arr;
    EXPECT_TRUE(result.shape().empty());
}
