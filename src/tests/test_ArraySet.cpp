/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-02-26
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
#include "ds/ArraySet.h"


TEST(ArraySetTest, DefaultConstructor) {
	ArraySet<int> s;
	EXPECT_EQ(s.size(), 0);
}

TEST(ArraySetTest, CapacityConstructor) {
	ArraySet<int> s(10);
	EXPECT_EQ(s.size(), 0);
}

TEST(ArraySetTest, ArrayConstructor) {
	int arr[] = {5, 3, 4, 3};
	ArraySet<int> s(arr, 4);
	EXPECT_EQ(s.size(), 3);
	EXPECT_EQ(s.get(0), 3);
	EXPECT_EQ(s.get(1), 4);
	EXPECT_EQ(s.get(2), 5);
}

TEST(ArraySetTest, Add) {
	ArraySet<int> s;
	s.add(5);
	EXPECT_EQ(s.size(), 1);
	EXPECT_EQ(s.get(0), 5);
	s.add(3);
	EXPECT_EQ(s.size(), 2);
	EXPECT_EQ(s.get(0), 3);
	EXPECT_EQ(s.get(1), 5);
	s.add(5); // duplicate, should not add
	EXPECT_EQ(s.size(), 2);
	s.add(4);
	EXPECT_EQ(s.size(), 3);
	EXPECT_EQ(s.get(0), 3);
	EXPECT_EQ(s.get(1), 4);
	EXPECT_EQ(s.get(2), 5);
}

TEST(ArraySetTest, AddManyArray) {
	ArraySet<int> s;
	int arr[] = {5, 3, 4, 3};
	s.addMany(arr, 4);
	EXPECT_EQ(s.size(), 3);
	EXPECT_EQ(s.get(0), 3);
	EXPECT_EQ(s.get(1), 4);
	EXPECT_EQ(s.get(2), 5);
}

TEST(ArraySetTest, Pop) {
	ArraySet<int> s;
	s.add(1);
	s.add(2);
	s.add(3);
	EXPECT_EQ(s.pop(), 3);
	EXPECT_EQ(s.size(), 2);
	EXPECT_EQ(s.get(1), 2);
}

TEST(ArraySetTest, Search) {
	ArraySet<int> s;
	EXPECT_EQ(s.search(1), -1);
	EXPECT_EQ(s.search(1, true), 0);
	s.add(1);
	s.add(3);
	s.add(5);
	EXPECT_EQ(s.search(3), 1);
	EXPECT_EQ(s.search(2), -1);
	EXPECT_EQ(s.search(2, true), 1); // insertion point
	EXPECT_EQ(s.search(0, true), 0);
	EXPECT_EQ(s.search(6, true), 3);
	EXPECT_EQ(s.search(5, true), 2); // exact, but returnNearest returns insertion which for exact is position, but wait, search returns position if found, else insertion point if returnNearest
}

TEST(ArraySetTest, Contains) {
	ArraySet<int> s;
	s.add(1);
	s.add(2);
	EXPECT_TRUE(s.contains(1));
	EXPECT_TRUE(s.contains(2));
	EXPECT_FALSE(s.contains(3));
}

TEST(ArraySetTest, Remove) {
	ArraySet<int> s;
	s.add(1);
	s.add(3);
	s.add(2);
	s.remove(2); // remove value 2
	EXPECT_EQ(s.size(), 2);
	EXPECT_EQ(s.get(0), 1);
	EXPECT_EQ(s.get(1), 3);
	s.removeAt(0); // remove index 0 (value 1)
	EXPECT_EQ(s.size(), 1);
	EXPECT_EQ(s.get(0), 3);
}

TEST(ArraySetTest, Prepare) {
	ArraySet<int> s;
	bool res = s.prepare(100);
	EXPECT_TRUE(res);
	// Further adds should work without failure, but can't check allocation directly
	for (int i = 0; i < 50; ++i) {
		s.add(i);
	}
	EXPECT_EQ(s.size(), 50);
}

TEST(ArraySetTest, Clear) {
	ArraySet<int> s;
	s.add(1);
	s.add(2);
	s.clear();
	EXPECT_EQ(s.size(), 0);
}

TEST(ArraySetTest, MinMax) {
	ArraySet<int> s;
	s.add(3);
	s.add(1);
	s.add(5);
	EXPECT_EQ(s.minimum(), 1);
	EXPECT_EQ(s.maximum(), 5);
}

TEST(ArraySetTest, BeginEnd) {
	ArraySet<int> s;
	s.add(1);
	s.add(2);
	int* it = s.begin();
	EXPECT_EQ(*it, 1);
	++it;
	EXPECT_EQ(*it, 2);
	++it;
	EXPECT_EQ(it, s.end());
	const ArraySet<int>& cs = s;
	const int* cit = cs.begin();
	EXPECT_EQ(*cit, 1);
	++cit;
	EXPECT_EQ(*cit, 2);
	++cit;
	EXPECT_EQ(cit, cs.end());
}

TEST(ArraySetTest, Growth) {
	ArraySet<int> s(1);
	for (int i = 0; i < 100; ++i) {
		s.add(i);
	}
	EXPECT_EQ(s.size(), 100);
	EXPECT_EQ(s.get(0), 0);
	EXPECT_EQ(s.get(99), 99);
}

TEST(ArraySetTest, GetMemory) {
	ArraySet<int> s;
	s.add(1);
	int* mem = s.getMemory();
	EXPECT_EQ(mem[0], 1);
}

TEST(ArraySetTest, ReverseIteration) {
    int arr[] = {1, 2, 3};
    ArraySet<int> set(arr, 3);
    std::vector<int> expected{3, 2, 1};
    
    int count = 0;
    for (auto it = set.rbegin(); it != set.rend(); ++it) {
        EXPECT_EQ(*it, expected[count]);
        count++;
    }
    EXPECT_EQ(count, 3);
}

TEST(ArraySetTest, ConstReverseIteration) {
    int arr[] = {1, 2, 3};
    ArraySet<int> set(arr, 3);
    const ArraySet<int>& cset = set;
    
    ArrayList<int> expected{3, 2, 1};
    int count = 0;
    for (auto it = cset.crbegin(); it != cset.crend(); ++it) {
        EXPECT_EQ(*it, expected.get(count));
        count++;
    }
    EXPECT_EQ(count, 3);
}
