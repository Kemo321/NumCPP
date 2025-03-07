#include "gtest/gtest.h"
#include "NDArray.tpp"


TEST(NDArrayTest, CopyEmptyArray) {
    NumCPP::NDArray<double> arr;
    NumCPP::NDArray<double> copy = arr.copy();
    EXPECT_EQ(copy.size(), 0);
    EXPECT_EQ(copy.shape(), std::vector<size_t>{});
}

TEST(NDArrayTest, Copy1DArray) {
    NumCPP::NDArray<double> arr({3}, std::vector<double>{1.0, 2.0, 3.0});
    NumCPP::NDArray<double> copy = arr.copy();
    EXPECT_EQ(copy.shape(), std::vector<size_t>{3});
    EXPECT_EQ(copy.strides(), arr.strides());
    EXPECT_EQ(copy(0), 1.0);
    EXPECT_EQ(copy(1), 2.0);
    EXPECT_EQ(copy(2), 3.0);
}

TEST(NDArrayTest, Copy2DArray) {
    NumCPP::NDArray<double> arr({2, 2}, std::vector<double>{1.0, 2.0, 3.0, 4.0});
    NumCPP::NDArray<double> copy = arr.copy();
    EXPECT_EQ(copy.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_EQ(copy.strides(), arr.strides());
    EXPECT_EQ(copy({0, 0}), 1.0);
    EXPECT_EQ(copy({0, 1}), 2.0);
    EXPECT_EQ(copy({1, 0}), 3.0);
    EXPECT_EQ(copy({1, 1}), 4.0);
}

TEST(NDArrayTest, CopyAndModifyOriginal) {
    NumCPP::NDArray<double> arr({2}, std::vector<double>{1.0, 2.0});
    NumCPP::NDArray<double> copy = arr.copy();
    arr(0) = 10.0;
    EXPECT_EQ(copy(0), 1.0);
    EXPECT_EQ(copy(1), 2.0);
}

TEST(NDArrayTest, Access1DWithFlatIndex) {
    NumCPP::NDArray<double> arr({3}, std::vector<double>{1.0, 2.0, 3.0});
    EXPECT_EQ(arr(0), 1.0);
    EXPECT_EQ(arr(1), 2.0);
    EXPECT_EQ(arr(2), 3.0);
}

TEST(NDArrayTest, Access2DWithIndices) {
    NumCPP::NDArray<double> arr({2, 2}, std::vector<double>{1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr({0, 0}), 1.0);
    EXPECT_EQ(arr({0, 1}), 2.0);
    EXPECT_EQ(arr({1, 0}), 3.0);
    EXPECT_EQ(arr({1, 1}), 4.0);
}

TEST(NDArrayTest, AccessWithOperatorSquareBrackets) {
    NumCPP::NDArray<double> arr({3}, std::vector<double>{1.0, 2.0, 3.0});
    EXPECT_EQ(arr[0], 1.0);
    EXPECT_EQ(arr[1], 2.0);
    EXPECT_EQ(arr[2], 3.0);

    NumCPP::NDArray<double> arr2d({2, 2}, std::vector<double>{1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(arr2d[std::vector<size_t>({0, 0})], 1.0);
    EXPECT_EQ(arr2d[std::vector<size_t>({0, 1})], 2.0);
    EXPECT_EQ(arr2d[std::vector<size_t>({1, 0})], 3.0);
    EXPECT_EQ(arr2d[std::vector<size_t>({1, 1})], 4.0);
}

TEST(NDArrayTest, ConstAccess) {
    const NumCPP::NDArray<double> arr({2}, std::vector<double>{1.0, 2.0});
    EXPECT_EQ(arr(0), 1.0);
    EXPECT_EQ(arr[0], 1.0);
    // arr(0) = 3.0;  // Would not compile due to const correctness
}

TEST(NDArrayTest, ModifyElement) {
    NumCPP::NDArray<double> arr({2}, std::vector<double>{1.0, 2.0});
    arr(0) = 10.0;
    EXPECT_EQ(arr(0), 10.0);
    EXPECT_EQ(arr(1), 2.0);

    arr[1] = 20.0;
    EXPECT_EQ(arr[1], 20.0);
}