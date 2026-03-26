//
// Created by komozoi on 3.3.26.
//

#include <ds/ArrayList.h>
#include <gtest/gtest.h>

TEST(ArrayListTest, DefaultConstruction) {
	ArrayList<int> list;
	EXPECT_EQ(list.size(), 0);
}

TEST(ArrayListTest, SingleValueConstructorCopy) {
	ArrayList<float> list(42.0f);
	ASSERT_EQ(list.size(), 1);
	EXPECT_EQ(list.get(0), 42.0f);
}

TEST(ArrayListTest, InitializerListConstructor) {
	ArrayList<int> list{1,2,3,4};
	ASSERT_EQ(list.size(), 4);
	EXPECT_EQ(list.get(0), 1);
	EXPECT_EQ(list.get(3), 4);
}

TEST(ArrayListTest, AddAndGrow) {
	ArrayList<int> list;
	list.add(1);
	for (int i = 2; i <= 200; i++)
		list.add(i);

	ASSERT_EQ(list.size(), 200);
	EXPECT_EQ(list.get(0), 1);
	EXPECT_EQ(list.get(199), 200);
}

TEST(ArrayListTest, AddRValue) {
	ArrayList<std::string> list;
	std::string s = "hello";
	list.add(std::move(s));
	ASSERT_EQ(list.size(), 1);
	EXPECT_EQ(list.get(0), "hello");
}

TEST(ArrayListTest, AddFirst) {
	ArrayList<int> list{2,3,4};
	list.addFirst(1);
	ASSERT_EQ(list.size(), 4);
	EXPECT_EQ(list.get(0), 1);
	EXPECT_EQ(list.get(1), 2);
}

TEST(ArrayListTest, AddManyPointer) {
	int vals[] = {5,6,7};
	ArrayList<int> list{1,2,3,4};
	list.addMany(vals, 3);

	ASSERT_EQ(list.size(), 7);
	EXPECT_EQ(list.get(4), 5);
	EXPECT_EQ(list.get(6), 7);
}

TEST(ArrayListTest, AddManyArrayList) {
	ArrayList<int> a{1,2};
	ArrayList<int> b{3,4};
	a.addMany(b);

	ASSERT_EQ(a.size(), 4);
	EXPECT_EQ(a.get(2), 3);
	EXPECT_EQ(a.get(3), 4);
}

TEST(ArrayListTest, AddCopies) {
	ArrayList<int> list;
	list.addCopies(9, 5);

	ASSERT_EQ(list.size(), 5);
	for (int i = 0; i < 5; i++)
		EXPECT_EQ(list.get(i), 9);
}

TEST(ArrayListTest, SetAndGet) {
	ArrayList<int> list{1,2,3};
	list.set(1, 42);
	EXPECT_EQ(list.get(1), 42);
}

TEST(ArrayListTest, Pop) {
	ArrayList<int> list{1,2,3};
	int v = list.pop();
	EXPECT_EQ(v, 3);
	EXPECT_EQ(list.size(), 2);
}

TEST(ArrayListTest, UnorderedRemove) {
	ArrayList<int> list{10,20,30,40};
	list.unorderedRemove(1);

	ASSERT_EQ(list.size(), 3);
	EXPECT_NE(list.get(1), 20);
}

TEST(ArrayListTest, ResizeGrowZeroFill) {
	ArrayList<int> list{1,2};
	list.resize(5);

	ASSERT_EQ(list.size(), 5);
	EXPECT_EQ(list.get(0), 1);
	EXPECT_EQ(list.get(1), 2);
	EXPECT_EQ(list.get(2), 0);
	EXPECT_EQ(list.get(4), 0);
}

TEST(ArrayListTest, ResizeShrink) {
	ArrayList<int> list{1,2,3,4};
	list.resize(2);
	ASSERT_EQ(list.size(), 2);
	EXPECT_EQ(list.get(0), 1);
	EXPECT_EQ(list.get(1), 2);
}

TEST(ArrayListTest, PrepareIncreaseCapacity) {
	ArrayList<int> list;
	bool ok = list.prepare(128);
	EXPECT_TRUE(ok);
	list.add(1);
	EXPECT_EQ(list.size(), 1);
}

TEST(ArrayListTest, SubscriptInPlacePositiveRange) {
	ArrayList<int> list{1,2,3,4,5};
	auto sub = list.subscriptInPlace(1,4);

	ASSERT_EQ(sub.size(), 3);
	EXPECT_EQ(sub.get(0), 2);
	EXPECT_EQ(sub.get(2), 4);
}

TEST(ArrayListTest, SubscriptInPlaceNegativeIndices) {
	ArrayList<int> list{1,2,3,4,5};
	auto sub = list.subscriptInPlace(-4,-1);

	ASSERT_EQ(sub.size(), 3);
	EXPECT_EQ(sub.get(0), 2);
	EXPECT_EQ(sub.get(2), 4);
}

TEST(ArrayListTest, CopyConstructorDeepCopy) {
	ArrayList<int> a{1,2,3};
	ArrayList<int> b(a);

	ASSERT_EQ(b.size(), 3);
	b.set(0, 99);

	EXPECT_EQ(a.get(0), 1);
	EXPECT_EQ(b.get(0), 99);
}

TEST(ArrayListTest, MoveConstructorTransfersOwnership) {
	ArrayList<int> a{1,2,3};
	ArrayList<int> b(std::move(a));

	ASSERT_EQ(b.size(), 3);
	EXPECT_EQ(b.get(0), 1);
	EXPECT_EQ(a.size(), 0);
}

TEST(ArrayListTest, MoveAssignment) {
	ArrayList<int> a{1,2,3};
	ArrayList<int> b;
	b = std::move(a);

	ASSERT_EQ(b.size(), 3);
	EXPECT_EQ(b.get(2), 3);
	EXPECT_EQ(a.size(), 0);
}

TEST(ArrayListTest, CopyAssignment) {
	ArrayList<int> a{7,8,9};
	ArrayList<int> b;
	b = a;

	ASSERT_EQ(b.size(), 3);
	EXPECT_EQ(b.get(1), 8);

	b.set(1, 42);
	EXPECT_EQ(a.get(1), 8);
}

TEST(ArrayListTest, ExportMemoryTransfersOwnership) {
	ArrayList<int> list{1,2,3};
	int sizeBefore = list.size();
	EXPECT_EQ(sizeBefore, 3);

	int* mem = list.exportMemory();
	EXPECT_EQ(list.size(), 0);
	EXPECT_NE(mem, nullptr);

	free(mem);  // caller now owns memory
}
