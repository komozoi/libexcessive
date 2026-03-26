//
// Created by komozoi on 15.8.25.
//
#include "gtest/gtest.h"

#include "ds/ShortQueue.h"


TEST(ShortQueue, InsertMaintainsSortedOrder) {
	ShortQueue<const char*, int> q(5);
	q.insert("low", 1);
	q.insert("medium", 3);
	q.insert("high", 2);

	EXPECT_STREQ(q.getMin(), "low");
	EXPECT_EQ(q.size(), 3);

	// Deletion should follow sorted order: low (1), high (2), medium (3)
	const char* val = q.deleteMin();
	EXPECT_STREQ(val, "low");

	val = q.deleteMin();
	EXPECT_STREQ(val, "high");

	val = q.deleteMin();
	EXPECT_STREQ(val, "medium");

	EXPECT_TRUE(q.isEmpty());
}

TEST(ShortQueue, InsertDropsHighestWhenFullAndNewIsBetter) {
	ShortQueue<const char*, int> q(3);
	q.insert("a", 5);  // worst
	q.insert("b", 3);
	q.insert("c", 1);  // best

	// Now full: [c (1), b (3), a (5)]

	// Insert "d" with priority 2. Should drop "a" (5)
	q.insert("d", 2);

	// Remaining: c (1), d (2), b (3)
	EXPECT_EQ(q.size(), 3);
	EXPECT_STREQ(q.getMin(), "c");

	EXPECT_STREQ(q.deleteMin(), "c");
	EXPECT_STREQ(q.deleteMin(), "d");
	EXPECT_STREQ(q.deleteMin(), "b");

	EXPECT_TRUE(q.isEmpty());
}

TEST(ShortQueue, InsertIgnoredIfNewItemWorseThanWorstAndFull) {
	ShortQueue<const char*, int> q(2);
	q.insert("b", 3);
	q.insert("a", 2);

	// Try inserting something worse
	q.insert("c", 4);

	EXPECT_EQ(q.size(), 2);
	EXPECT_STREQ(q.deleteMin(), "a");
	EXPECT_STREQ(q.deleteMin(), "b");
	EXPECT_TRUE(q.isEmpty());
}

TEST(ShortQueue, DeleteMinReturnsFalseWhenEmpty) {
	ShortQueue<int, int> q(2);

	int valOut = -1;
	int priOut = -1;

	EXPECT_FALSE(q.deleteMin(&valOut, &priOut));
}

TEST(ShortQueue, DeleteMinOutputsValueAndPriorityCorrectly) {
	ShortQueue<int, int> q(2);
	q.insert(100, 7);
	q.insert(200, 3);

	int valOut = 0;
	int priOut = 0;
	EXPECT_TRUE(q.deleteMin(&valOut, &priOut));
	EXPECT_EQ(valOut, 200);
	EXPECT_EQ(priOut, 3);

	EXPECT_TRUE(q.deleteMin(&valOut, &priOut));
	EXPECT_EQ(valOut, 100);
	EXPECT_EQ(priOut, 7);
}

TEST(ShortQueue, InsertHandlesDescendingPriorities) {
	ShortQueue<int, int> q(4);
	q.insert(1, 4);
	q.insert(2, 3);
	q.insert(3, 2);
	q.insert(4, 1);

	// Should now be sorted: 4(1), 3(2), 2(3), 1(4)
	EXPECT_EQ(q.deleteMin(), 4);
	EXPECT_EQ(q.deleteMin(), 3);
	EXPECT_EQ(q.deleteMin(), 2);
	EXPECT_EQ(q.deleteMin(), 1);
}

TEST(ShortQueue, IsEmptyAndSizeWorkCorrectly) {
	ShortQueue<int, int> q(1);
	EXPECT_TRUE(q.isEmpty());
	EXPECT_EQ(q.size(), 0);

	q.insert(42, 1);
	EXPECT_FALSE(q.isEmpty());
	EXPECT_EQ(q.size(), 1);

	q.deleteMin();
	EXPECT_TRUE(q.isEmpty());
	EXPECT_EQ(q.size(), 0);
}