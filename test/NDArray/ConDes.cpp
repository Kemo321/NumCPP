#include <gtest/gtest.h>
#include "NDArray.hpp"  // Include your NDArray header

namespace NumCPP {

////////////////////////////////////////////////////////////////
// Tests for Default Constructor
////////////////////////////////////////////////////////////////

TEST(NDArrayTest, DefaultConstructor) {
    NDArray<int> array;
    EXPECT_EQ(array.size(), 0);  // The array should have no elements
    EXPECT_EQ(array.shape().size(), 0);  // Shape of the array should be empty
}

TEST(NDArrayTest, DefaultConstructor2) {
    NDArray<double> array;
    EXPECT_EQ(array.size(), 0);  // The array should have no elements
    EXPECT_EQ(array.shape().size(), 0);  // Shape of the array should be empty
}

TEST(NDArrayTest, DefaultConstructor3) {
    NDArray<float> array;
    EXPECT_EQ(array.size(), 0);  // The array should have no elements
    EXPECT_EQ(array.shape().size(), 0);  // Shape of the array should be empty
}

TEST(NDArrayTest, DefaultConstructor4) {
    NDArray<char> array;
    EXPECT_EQ(array.size(), 0);  // The array should have no elements
    EXPECT_EQ(array.shape().size(), 0);  // Shape of the array should be empty
}


////////////////////////////////////////////////////////////////
// Tests for Destructor
////////////////////////////////////////////////////////////////

TEST(NDArrayTest, Destructor) {
    NDArray<int>* array = new NDArray<int>();  // Create array dynamically
    EXPECT_NO_THROW(delete array);  // Destructor should not throw any exceptions
}

TEST(NDArrayTest, Destructor2) {
    NDArray<double>* array = new NDArray<double>();  // Create array dynamically
    EXPECT_NO_THROW(delete array);  // Destructor should not throw any exceptions
}

TEST(NDArrayTest, Destructor3) {
    NDArray<float>* array = new NDArray<float>();  // Create array dynamically
    EXPECT_NO_THROW(delete array);  // Destructor should not throw any exceptions
}

TEST(NDArrayTest, Destructor4) {
    NDArray<char>* array = new NDArray<char>();  // Create array dynamically
    EXPECT_NO_THROW(delete array);  // Destructor should not throw any exceptions
}

////////////////////////////////////////////////////////////////
// Test for Copy Constructor
////////////////////////////////////////////////////////////////

TEST(NDArrayTest, CopyConstructor) {
    NDArray<int> array1({2, 2}, 3);  // Create a 2x2 array initialized with 3
    NDArray<int> array2 = array1;  // Use copy constructor

    EXPECT_EQ(array2.size(), 4);  // array2 should have 4 elements (2x2)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 3);  // All elements of array2 should be 3
    }
}

TEST(NDArrayTest, CopyConstructor2) {
    NDArray<double> array1({4, 12}, 3);  // Create a 2x2 array initialized with 3
    NDArray<double> array2 = array1;  // Use copy constructor

    EXPECT_EQ(array2.size(), 48);  // array2 should have 4 elements (2x2)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 3);  // All elements of array2 should be 3
    }
}

TEST(NDArrayTest, CopyConstructor3) {
    NDArray<float> array1({10, 2}, 3);  // Create a 2x2 array initialized with 3
    NDArray<float> array2 = array1;  // Use copy constructor

    EXPECT_EQ(array2.size(), 20);  // array2 should have 4 elements (2x2)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 3);  // All elements of array2 should be 3
    }
}

TEST(NDArrayTest, CopyConstructor4) {
    NDArray<char> array1({2, 2}, 3);  // Create a 2x2 array initialized with 3
    NDArray<char> array2 = array1;  // Use copy constructor

    EXPECT_EQ(array2.size(), 4);  // array2 should have 4 elements (2x2)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 3);  // All elements of array2 should be 3
    }
}

////////////////////////////////////////////////////////////////
// Test for Move Constructor
////////////////////////////////////////////////////////////////

TEST(NDArrayTest, MoveConstructor) {
    NDArray<int> array1({2, 2}, 4);  // Create a 2x2 array initialized with 4
    NDArray<int> array2 = std::move(array1);  // Move array1 to array2

    EXPECT_EQ(array2.size(), 4);  // array2 should have the size of array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 4);  // All elements of array2 should be 4
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

TEST(NDArrayTest, MoveConstructor2) {
    NDArray<double> array1({2, 2}, 4);  // Create a 2x2 array initialized with 4
    NDArray<double> array2 = std::move(array1);  // Move array1 to array2

    EXPECT_EQ(array2.size(), 4);  // array2 should have the size of array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 4);  // All elements of array2 should be 4
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

TEST(NDArrayTest, MoveConstructor3) {
    NDArray<float> array1({2, 2}, 4);  // Create a 2x2 array initialized with 4
    NDArray<float> array2 = std::move(array1);  // Move array1 to array2

    EXPECT_EQ(array2.size(), 4);  // array2 should have the size of array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 4);  // All elements of array2 should be 4
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

TEST(NDArrayTest, MoveConstructor4) {
    NDArray<char> array1({2, 2}, 4);  // Create a 2x2 array initialized with 4
    NDArray<char> array2 = std::move(array1);  // Move array1 to array2

    EXPECT_EQ(array2.size(), 4);  // array2 should have the size of array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 4);  // All elements of array2 should be 4
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

////////////////////////////////////////////////////////////////
// Test for Copy Assignment Operator
////////////////////////////////////////////////////////////////

TEST(NDArrayTest, CopyAssignmentOperator) {
    NDArray<int> array1({3, 3}, 2);  // Create a 3x3 array initialized with 2
    NDArray<int> array2({2, 2}, 5);  // Create another 2x2 array initialized with 5

    array2 = array1;  // Use copy assignment operator

    EXPECT_EQ(array2.size(), 9);  // After assignment, array2 should be the same size as array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 2);  // All elements of array2 should be 2 (assigned from array1)
    }
}

TEST(NDArrayTest, CopyAssignmentOperator2) {
    NDArray<double> array1({3, 3}, 2);  // Create a 3x3 array initialized with 2
    NDArray<double> array2({2, 2}, 5);  // Create another 2x2 array initialized with 5

    array2 = array1;  // Use copy assignment operator

    EXPECT_EQ(array2.size(), 9);  // After assignment, array2 should be the same size as array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 2);  // All elements of array2 should be 2 (assigned from array1)
    }
}

TEST(NDArrayTest, CopyAssignmentOperator3) {
    NDArray<float> array1({3, 3}, 2);  // Create a 3x3 array initialized with 2
    NDArray<float> array2({2, 2}, 5);  // Create another 2x2 array initialized with 5

    array2 = array1;  // Use copy assignment operator

    EXPECT_EQ(array2.size(), 9);  // After assignment, array2 should be the same size as array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 2);  // All elements of array2 should be 2 (assigned from array1)
    }
}

TEST(NDArrayTest, CopyAssignmentOperator4) {
    NDArray<char> array1({3, 3}, 2);  // Create a 3x3 array initialized with 2
    NDArray<char> array2({2, 2}, 5);  // Create another 2x2 array initialized with 5

    array2 = array1;  // Use copy assignment operator

    EXPECT_EQ(array2.size(), 9);  // After assignment, array2 should be the same size as array1
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 2);  // All elements of array2 should be 2 (assigned from array1)
    }
}

////////////////////////////////////////////////////////////////
// Test for Move Assignment Operator
////////////////////////////////////////////////////////////////

TEST(NDArrayTest, MoveAssignmentOperator) {
    NDArray<int> array1({3, 3}, 1);  // Create a 3x3 array initialized with 1
    NDArray<int> array2({2, 2}, 10);  // Create another 2x2 array initialized with 10

    array2 = std::move(array1);  // Move array1 into array2

    EXPECT_EQ(array2.size(), 9);  // array2 should have the size of array1 (3x3)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 1);  // All elements of array2 should be 1 (moved from array1)
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

TEST(NDArrayTest, MoveAssignmentOperator2) {
    NDArray<double> array1({3, 3}, 1);  // Create a 3x3 array initialized with 1
    NDArray<double> array2({2, 2}, 10);  // Create another 2x2 array initialized with 10

    array2 = std::move(array1);  // Move array1 into array2

    EXPECT_EQ(array2.size(), 9);  // array2 should have the size of array1 (3x3)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 1);  // All elements of array2 should be 1 (moved from array1)
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

TEST(NDArrayTest, MoveAssignmentOperator3) {
    NDArray<float> array1({3, 3}, 1);  // Create a 3x3 array initialized with 1
    NDArray<float> array2({2, 2}, 10);  // Create another 2x2 array initialized with 10

    array2 = std::move(array1);  // Move array1 into array2

    EXPECT_EQ(array2.size(), 9);  // array2 should have the size of array1 (3x3)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 1);  // All elements of array2 should be 1 (moved from array1)
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

TEST(NDArrayTest, MoveAssignmentOperator4) {
    NDArray<char> array1({3, 3}, 1);  // Create a 3x3 array initialized with 1
    NDArray<char> array2({2, 2}, 10);  // Create another 2x2 array initialized with 10

    array2 = std::move(array1);  // Move array1 into array2

    EXPECT_EQ(array2.size(), 9);  // array2 should have the size of array1 (3x3)
    for (size_t i = 0; i < array2.size(); ++i) {
        EXPECT_EQ(array2(static_cast<int>(i)), 1);  // All elements of array2 should be 1 (moved from array1)
    }
    EXPECT_EQ(array1.size(), 0);  // After move, array1 should be in a valid but unspecified state
}

////////////////////////////////////////////////////////////////
// Test for Constructor with Shape and Initialization Value
////////////////////////////////////////////////////////////////

TEST(NDArrayTest, ConstructorWithShapeAndInitVal) {
    NDArray<int> array({3, 3}, 5);  // Create a 3x3 array, all elements initialized to 5

    EXPECT_EQ(array.size(), 9);  // Total elements should be 9 (3x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 5);  // Ensure all elements are initialized to 5
    }
}

TEST(NDArrayTest, ConstructorWithShapeAndInitVal2) {
    NDArray<double> array({3, 3}, 5);  // Create a 3x3 array, all elements initialized to 5

    EXPECT_EQ(array.size(), 9);  // Total elements should be 9 (3x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 5);  // Ensure all elements are initialized to 5
    }
}

TEST(NDArrayTest, ConstructorWithShapeAndInitVal3) {
    NDArray<float> array({3, 3}, 5);  // Create a 3x3 array, all elements initialized to 5

    EXPECT_EQ(array.size(), 9);  // Total elements should be 9 (3x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 5);  // Ensure all elements are initialized to 5
    }
}

TEST(NDArrayTest, ConstructorWithShapeAndInitVal4) {
    NDArray<char> array({3, 3}, 5);  // Create a 3x3 array, all elements initialized to 5

    EXPECT_EQ(array.size(), 9);  // Total elements should be 9 (3x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 5);  // Ensure all elements are initialized to 5
    }
}

TEST(NDArray, InvalidShape) {
    EXPECT_THROW(NDArray<int>({3, 0}, 5), std::invalid_argument);
}

TEST(NDArray, InvalidShape2) {
    EXPECT_THROW(NDArray<double>({3, 0}, 5), std::invalid_argument);
}

TEST(NDArray, InvalidShape3) {
    EXPECT_THROW(NDArray<float>({1, 0}, 5), std::invalid_argument);
}

////////////////////////////////////////////////////////////////
// Test for Constructor with Initializer List
////////////////////////////////////////////////////////////////
TEST(NDArrayTest, ConstructorWithInitializerList) {
    NDArray<int> array({2, 3}, 7);  // Create a 2x3 array, all elements initialized to 7

    EXPECT_EQ(array.size(), 6);  // Total elements should be 6 (2x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 7);  // Ensure all elements are initialized to 7
    }
}

TEST(NDArrayTest, ConstructorWithInitializerList2) {
    NDArray<double> array({2, 3}, 7);  // Create a 2x3 array, all elements initialized to 7

    EXPECT_EQ(array.size(), 6);  // Total elements should be 6 (2x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 7);  // Ensure all elements are initialized to 7
    }
}

TEST(NDArrayTest, ConstructorWithInitializerList3) {
    NDArray<float> array({2, 3}, 7);  // Create a 2x3 array, all elements initialized to 7

    EXPECT_EQ(array.size(), 6);  // Total elements should be 6 (2x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 7);  // Ensure all elements are initialized to 7
    }

}

TEST(NDArrayTest, ConstructorWithInitializerList4) {
    NDArray<char> array({2, 3}, 7);  // Create a 2x3 array, all elements initialized to 7

    EXPECT_EQ(array.size(), 6);  // Total elements should be 6 (2x3)
    for (size_t i = 0; i < array.size(); ++i) {
        EXPECT_EQ(array(static_cast<int>(i)), 7);  // Ensure all elements are initialized to 7
    }
}

}  // namespace NumCPP
