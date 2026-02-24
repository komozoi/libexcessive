//
// Created by nuclaer on 19.2.25.
//

#include "alloc/SlabAllocator.h"
#include <gtest/gtest.h>
#include <memory>


typedef struct {
	int a, b, c, d;
} test_struct_t;


TEST(SlabPointerTest, BasicUsage) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		// Create a new SlabPointer with an integer value
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));

		// Check if the pointer is valid
		EXPECT_TRUE(ptr.get() != nullptr);

		// Copy the pointer and check reference count
		SlabPointer<test_struct_t> copy_ptr = ptr;
		EXPECT_EQ(copy_ptr.get(), ptr.get());

		// Increment the reference count
		copy_ptr.reset(slab.allocate<test_struct_t>({4, 3, 2, 1}));

		// Check if the original pointer is still valid
		EXPECT_TRUE(ptr.get() != nullptr);
	}

	// Pointers went out of scope, so we should be back where we started
	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, CopyConstructor) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		SlabPointer<test_struct_t> copy_ptr(ptr);

		EXPECT_NE(copy_ptr.get(), nullptr);
		EXPECT_EQ(copy_ptr.get(), ptr.get());
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, MoveConstructor) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		SlabPointer<test_struct_t> moved_ptr(std::move(ptr));

		EXPECT_NE(moved_ptr.get(), nullptr);
		EXPECT_EQ(ptr.get(), nullptr);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, Reset) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		ptr.reset(slab.allocate<test_struct_t>({5, 6, 7, 8}));

		EXPECT_NE(ptr.get(), nullptr);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, MultipleReferences) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		SlabPointer<test_struct_t> copy_ptr(ptr);

		EXPECT_NE(copy_ptr.get(), nullptr);
		EXPECT_EQ(copy_ptr.get(), ptr.get());
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

/*TEST(SlabPointerTest, SelfReset) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		ptr.reset(ptr.release());

		EXPECT_EQ(ptr.get(), nullptr);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, Release) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		test_struct_t* raw_ptr = ptr.release();

		EXPECT_EQ(ptr.get(), nullptr);
		EXPECT_NE(raw_ptr, nullptr);

		// Manually deallocate to avoid memory leak
		slab.free(raw_ptr);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}*/

TEST(SlabPointerTest, DefaultConstructor) {
	SlabPointer<test_struct_t> ptr;

	EXPECT_EQ(ptr.get(), nullptr);
}

TEST(SlabPointerTest, AssignOperator) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		SlabPointer<test_struct_t> copy_ptr;
		copy_ptr = ptr;

		EXPECT_NE(copy_ptr.get(), nullptr);
		EXPECT_EQ(copy_ptr.get(), ptr.get());
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, MoveAssign) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		SlabPointer<test_struct_t> moved_ptr;
		moved_ptr = std::move(ptr);

		EXPECT_NE(moved_ptr.get(), nullptr);
		EXPECT_EQ(ptr.get(), nullptr);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

/*TEST(SlabPointerTest, Swap) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr1(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		SlabPointer<test_struct_t> ptr2(slab.allocate<test_struct_t>({5, 6, 7, 8}));
		ptr1.swap(ptr2);

		EXPECT_NE(ptr1.get(), nullptr);
		EXPECT_NE(ptr2.get(), nullptr);
		EXPECT_NE(ptr1.get(), ptr2.get());
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}*/

TEST(SlabPointerTest, OperatorArrow) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		ptr->a = 5;

		EXPECT_EQ(ptr->a, 5);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, OperatorStar) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		*ptr = {5, 6, 7, 8};

		EXPECT_EQ(ptr->a, 5);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, Get) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr(slab.allocate<test_struct_t>({1, 2, 3, 4}));
		test_struct_t* raw_ptr = ptr.get();

		EXPECT_NE(raw_ptr, nullptr);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, MakePtr) {
	size_t initiallyAllocated = slab.getBytesAllocated();

	{
		SlabPointer<test_struct_t> ptr({1, 2, 3, 4});
		test_struct_t* raw_ptr = ptr.get();

		EXPECT_NE(raw_ptr, nullptr);
	}

	EXPECT_EQ(initiallyAllocated, slab.getBytesAllocated());
}

TEST(SlabPointerTest, NullPtrCheck) {
	SlabPointer<test_struct_t> ptr;

	EXPECT_EQ(ptr.get(), nullptr);
	EXPECT_FALSE(static_cast<bool>(ptr));
}
