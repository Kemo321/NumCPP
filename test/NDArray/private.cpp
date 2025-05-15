#include <gtest/gtest.h>
#include "Array.hpp"

using namespace NumCPP;

class PublicArray : public Array<double> {
public:
    using Array<double>::Array; // Inherit constructors
    using Array<double>::compute_index; // Expose compute_index
    using Array<double>::strides; // Expose strides
};

TEST(ComputeStridesAndIndex, ComputeStrides2D) {
    PublicArray arr({2, 3});
    EXPECT_EQ(arr.strides(), std::vector<size_t>({3, 1}));
}

TEST(ComputeStridesAndIndex, ComputeStrides3D) {
    PublicArray arr({2, 3, 4});
    EXPECT_EQ(arr.strides(), std::vector<size_t>({12, 4, 1}));
}

TEST(ComputeStridesAndIndex, ComputeIndex2D) {
    PublicArray arr({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    size_t index = arr.compute_index({1, 2});
    EXPECT_EQ(index, 5);
    EXPECT_EQ(arr(1, 2), 6.0);
}

TEST(ComputeStridesAndIndex, ComputeIndexOutOfBounds) {
    PublicArray arr({2, 2});
    EXPECT_THROW(arr.compute_index({2, 0}), std::runtime_error);
}