//
// Created by komozoi on 26.7.25.
//
// test_hashmap.cpp

#include "ds/HashMap.h"
#include <gtest/gtest.h>
#include <string>

TEST(HashMapTest, InsertAndRetrieve) {
	HashMap<int, std::string> map(8);
	map.put(1, "apple");
	map.put(2, "banana");

	EXPECT_TRUE(map.hasKey(1));
	EXPECT_TRUE(map.hasKey(2));
	EXPECT_FALSE(map.hasKey(3));

	EXPECT_EQ(map.get(1), "apple");
	EXPECT_EQ(map.get(2), "banana");

	std::string* ptr = map.getPtr(2);
	ASSERT_NE(ptr, nullptr);
	EXPECT_EQ(*ptr, "banana");
}

TEST(HashMapTest, OverwriteValue) {
	HashMap<int, std::string> map(4);
	map.put(42, "initial");
	map.put(42, "updated");

	EXPECT_EQ(map.get(42), "updated");
}

TEST(HashMapTest, RemoveEntry) {
	HashMap<int, std::string> map(4);
	map.put(7, "hello");
	EXPECT_TRUE(map.hasKey(7));

	std::string removed;
	EXPECT_TRUE(map.remove(7, removed));
	EXPECT_EQ(removed, "hello");
	EXPECT_FALSE(map.hasKey(7));
	EXPECT_EQ(map.numUsed(), 0);
}

TEST(HashMapTest, RemoveNonExistentKey) {
	HashMap<int, std::string> map(4);
	std::string out;
	EXPECT_FALSE(map.remove(123, out));
}

TEST(HashMapTest, ResizeOnFull) {
	HashMap<int, std::string> map(2);
	map.put(1, "one");
	map.put(2, "two");
	map.put(3, "three");  // Should trigger resize

	EXPECT_EQ(map.getCapacity(), 4);
	EXPECT_EQ(map.numUsed(), 3);
	EXPECT_EQ(map.get(3), "three");
}

TEST(HashMapTest, ClearMap) {
	HashMap<int, std::string> map(4);
	map.put(1, "one");
	map.put(2, "two");
	map.clear();

	EXPECT_EQ(map.numUsed(), 0);
	EXPECT_FALSE(map.hasKey(1));
	EXPECT_FALSE(map.hasKey(2));
}

TEST(HashMapTest, CopyConstructor) {
	HashMap<int, std::string> original(4);
	original.put(10, "ten");

	HashMap<int, std::string> copy = original;
	EXPECT_TRUE(copy.hasKey(10));
	EXPECT_EQ(copy.get(10), "ten");

	copy.put(10, "updated");
	EXPECT_EQ(original.get(10), "ten");  // ensure original not modified
}

TEST(HashMapTest, MoveConstructor) {
	HashMap<int, std::string> original(4);
	original.put(11, "eleven");

	HashMap<int, std::string> moved = std::move(original);
	EXPECT_TRUE(moved.hasKey(11));
	EXPECT_EQ(moved.get(11), "eleven");
}

TEST(HashMapTest, AssignmentOperators) {
	HashMap<int, std::string> a(2);
	a.put(5, "five");

	HashMap<int, std::string> b(4);
	b = a;
	EXPECT_EQ(b.get(5), "five");

	HashMap<int, std::string> c(1);
	c = std::move(b);
	EXPECT_EQ(c.get(5), "five");
}

TEST(HashMapTest, PairKeyInsertAndRetrieve) {
	HashMap<std::pair<int,int>, std::string> map(8);

	map.put({1,2}, "alpha");
	map.put({3,4}, "beta");

	EXPECT_TRUE(map.hasKey({1,2}));
	EXPECT_TRUE(map.hasKey({3,4}));
	EXPECT_FALSE(map.hasKey({5,6}));

	EXPECT_EQ(map.get({1,2}), "alpha");
	EXPECT_EQ(map.get({3,4}), "beta");

	std::string* ptr = map.getPtr({3,4});
	ASSERT_NE(ptr, nullptr);
	EXPECT_EQ(*ptr, "beta");
}
