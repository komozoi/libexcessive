/*
 * Copyright 2023-2025 komozoi
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include "ds/LinkedList.h"
#include "ds/ArrayList.h"

TEST(LinkedListTest, BasicOps) {
	LinkedList<int> list;
	EXPECT_EQ(list.size(), 0);

	list.add(10);
	list.add(20);
	list.add(30);

	EXPECT_EQ(list.size(), 3);
	EXPECT_EQ(list.get(0), 10);
	EXPECT_EQ(list.get(1), 20);
	EXPECT_EQ(list.get(2), 30);
}

TEST(LinkedListTest, Iteration) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	list.add(3);

	int count = 0;
	int expected[] = {1, 2, 3};

	// This is how it's currently done with cursor
	list.resetCursor();
	while (list.isCursorValid()) {
		EXPECT_EQ(list.next(), expected[count++]);
	}
	EXPECT_EQ(count, 3);
}

TEST(LinkedListTest, StandardIterator) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	list.add(3);

	int count = 0;
	int expected[] = {1, 2, 3};
	for (int val : list) {
		EXPECT_EQ(val, expected[count++]);
	}
	EXPECT_EQ(count, 3);
}

TEST(LinkedListTest, ConstIterator) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	list.add(3);

	const LinkedList<int>& clist = list;
	int count = 0;
	int expected[] = {1, 2, 3};
	for (int val : clist) {
		EXPECT_EQ(val, expected[count++]);
	}
	EXPECT_EQ(count, 3);
}

TEST(LinkedListTest, IteratorMutation) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	list.add(3);

	for (int& val : list) {
		val *= 10;
	}

	EXPECT_EQ(list.get(0), 10);
	EXPECT_EQ(list.get(1), 20);
	EXPECT_EQ(list.get(2), 30);
}

TEST(LinkedListTest, BidirectionalIterator) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	list.add(3);

	LinkedList<int>::Iterator it = list.begin();
	EXPECT_EQ(*it, 1);
	++it;
	EXPECT_EQ(*it, 2);
	--it;
	EXPECT_EQ(*it, 1);

	LinkedList<int>::Iterator it_end = list.end();
	--it_end;
	EXPECT_EQ(*it_end, 3);
}

TEST(LinkedListTest, ReverseIteration) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	list.add(3);

	int count = 0;
	int expected[] = {3, 2, 1};
	for (int& val : list.reverse()) {
		EXPECT_EQ(val, expected[count++]);
	}
	EXPECT_EQ(count, 3);
}

TEST(LinkedListTest, ConstReverseIteration) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	list.add(3);

	const LinkedList<int>& clist = list;
	int expected[] = {3, 2, 1};
	int count = 0;
	for (const int& val : clist.reverse()) {
		EXPECT_EQ(val, expected[count++]);
	}
	EXPECT_EQ(count, 3);
}

TEST(LinkedListTest, AddManyGeneric) {
	LinkedList<int> list;
	ArrayList<int> alist{1, 2, 3};
	list.addMany(alist);

	EXPECT_EQ(list.size(), 3);
	EXPECT_EQ(list.get(0), 1);
	EXPECT_EQ(list.get(1), 2);
	EXPECT_EQ(list.get(2), 3);
}

TEST(LinkedListTest, CopyConstructor) {
	LinkedList<int> list1;
	list1.add(1);
	list1.add(2);
	list1.add(3);

	LinkedList<int> list2 = list1;

	EXPECT_EQ(list2.size(), 3);
	EXPECT_EQ(list2.get(0), 1);
	EXPECT_EQ(list2.get(1), 2);
	EXPECT_EQ(list2.get(2), 3);

	// Ensure deep copy
	list1.set(0, 10);
	EXPECT_EQ(list1.get(0), 10);
	EXPECT_EQ(list2.get(0), 1);
}

TEST(LinkedListTest, MoveConstructor) {
	LinkedList<int> list1;
	list1.add(1);
	list1.add(2);

	LinkedList<int> list2 = std::move(list1);

	EXPECT_EQ(list2.size(), 2);
	EXPECT_EQ(list2.get(0), 1);
	EXPECT_EQ(list2.get(1), 2);
	EXPECT_EQ(list1.size(), 0);
}

TEST(LinkedListTest, CopyAssignment) {
	LinkedList<int> list1;
	list1.add(1);

	LinkedList<int> list2;
	list2.add(10);
	list2.add(20);

	list2 = list1;

	EXPECT_EQ(list2.size(), 1);
	EXPECT_EQ(list2.get(0), 1);

	// Ensure deep copy
	list1.set(0, 100);
	EXPECT_EQ(list2.get(0), 1);
}

TEST(LinkedListTest, MoveAssignment) {
	LinkedList<int> list1;
	list1.add(1);

	LinkedList<int> list2;
	list2.add(10);

	list2 = std::move(list1);

	EXPECT_EQ(list2.size(), 1);
	EXPECT_EQ(list2.get(0), 1);
	EXPECT_EQ(list1.size(), 0);
}

TEST(LinkedListTest, OutOfBoundsException) {
	LinkedList<int> list;
	list.add(1);
	list.add(2);
	EXPECT_THROW(list.get(-1), std::out_of_range);
	EXPECT_THROW(list.get(2), std::out_of_range);
}
