#include <gtest/gtest.h>
#include "Array.hpp"
#include <vector>

namespace NumCPP {

TEST(ArrayTest, ElementAccess1D) {
    Array<double> arr({3}, std::vector<double>{1, 2, 3});
    EXPECT_DOUBLE_EQ(arr(0), 1.0);
    EXPECT_DOUBLE_EQ(arr(1), 2.0);
    EXPECT_DOUBLE_EQ(arr(2), 3.0);
}

TEST(ArrayTest, ElementAccess2D) {
    Array<double> arr({2, 2}, std::vector<double>{1, 2, 3, 4});
    EXPECT_DOUBLE_EQ(arr({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(arr({0, 1}), 2.0);
    EXPECT_DOUBLE_EQ(arr({1, 0}), 3.0);
    EXPECT_DOUBLE_EQ(arr({1, 1}), 4.0);
}

TEST(ArrayTest, ElementAccessOutOfBounds) {
    Array<double> arr({2, 2}, std::vector<double>{1, 2, 3, 4});
    EXPECT_THROW(arr({2, 0}), std::runtime_error);
    EXPECT_THROW(arr({0, 2}), std::runtime_error);
}

} // namespace NumCPP