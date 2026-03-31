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
#include "ds/HashSet.h"


#include <gtest/gtest.h>

TEST(HashSetTest, ConstructorCapacity) {
	HashSet<int> s(10);
	EXPECT_EQ(s.getCapacity(), 10);
	EXPECT_EQ(s.size(), 0);
}

TEST(HashSetTest, ConstructorFromArray) {
	int arr[] = {1, 2, 3, 2};
	HashSet<int> s(arr, 4);
	EXPECT_EQ(s.size(), 3);
	EXPECT_TRUE(s.contains(1));
	EXPECT_TRUE(s.contains(2));
	EXPECT_TRUE(s.contains(3));
}

TEST(HashSetTest, Add) {
	HashSet<int> s(5);
	bool added = s.add(1);
	EXPECT_FALSE(added); // new
	EXPECT_EQ(s.size(), 1);
	EXPECT_TRUE(s.contains(1));
	added = s.add(1);
	EXPECT_TRUE(added); // already present
	EXPECT_EQ(s.size(), 1);
	s.add(2);
	s.add(3);
	EXPECT_EQ(s.size(), 3);
}

TEST(HashSetTest, Contains) {
	HashSet<int> s(5);
	s.add(1);
	s.add(3);
	EXPECT_TRUE(s.contains(1));
	EXPECT_FALSE(s.contains(2));
	EXPECT_TRUE(s.contains(3));
}

TEST(HashSetTest, Remove) {
	HashSet<int> s(5);
	s.add(1);
	s.add(2);
	s.add(3);
	bool removed = s.remove(2);
	EXPECT_TRUE(removed);
	EXPECT_EQ(s.size(), 2);
	EXPECT_FALSE(s.contains(2));
	EXPECT_TRUE(s.contains(1));
	EXPECT_TRUE(s.contains(3));
	removed = s.remove(4);
	EXPECT_FALSE(removed);
}

TEST(HashSetTest, Resize) {
	HashSet<int> s(2);
	s.add(1);
	s.add(2);
	EXPECT_EQ(s.size(), 2);
	EXPECT_EQ(s.getCapacity(), 2);
	s.add(3); // should resize to 4
	EXPECT_EQ(s.size(), 3);
	EXPECT_EQ(s.getCapacity(), 4);
	EXPECT_TRUE(s.contains(1));
	EXPECT_TRUE(s.contains(2));
	EXPECT_TRUE(s.contains(3));
}

TEST(HashSetTest, Clear) {
	HashSet<int> s(5);
	s.add(1);
	s.add(2);
	s.clear();
	EXPECT_EQ(s.size(), 0);
	EXPECT_FALSE(s.contains(1));
	EXPECT_FALSE(s.contains(2));
}

TEST(HashSetTest, CopyConstructor) {
	HashSet<int> s1(5);
	s1.add(1);
	s1.add(2);
	HashSet<int> s2(s1);
	EXPECT_EQ(s2.size(), 2);
	EXPECT_TRUE(s2.contains(1));
	EXPECT_TRUE(s2.contains(2));
	// Modify s1, s2 should not change
	s1.add(3);
	EXPECT_FALSE(s2.contains(3));
}

TEST(HashSetTest, CopyAssignment) {
	HashSet<int> s1(5);
	s1.add(1);
	s1.add(2);
	HashSet<int> s2(3);
	s2 = s1;
	EXPECT_EQ(s2.size(), 2);
	EXPECT_TRUE(s2.contains(1));
	EXPECT_TRUE(s2.contains(2));
	// Modify s1, s2 should not change
	s1.add(3);
	EXPECT_FALSE(s2.contains(3));
}

TEST(HashSetTest, MoveConstructor) {
	HashSet<int> s1(5);
	s1.add(1);
	s1.add(2);
	HashSet<int> s2(std::move(s1));
	EXPECT_EQ(s2.size(), 2);
	EXPECT_TRUE(s2.contains(1));
	EXPECT_TRUE(s2.contains(2));
	EXPECT_EQ(s1.size(), 0);
	EXPECT_EQ(s1.getCapacity(), 0);
}

TEST(HashSetTest, MoveAssignment) {
	HashSet<int> s1(5);
	s1.add(1);
	s1.add(2);
	HashSet<int> s2(3);
	s2 = std::move(s1);
	EXPECT_EQ(s2.size(), 2);
	EXPECT_TRUE(s2.contains(1));
	EXPECT_TRUE(s2.contains(2));
	EXPECT_EQ(s1.size(), 0);
	EXPECT_EQ(s1.getCapacity(), 0);
}

TEST(HashSetTest, AddFrom) {
	HashSet<int> s1(5);
	s1.add(1);
	s1.add(2);
	HashSet<int> s2(5);
	s2.addFrom(s1);
	EXPECT_EQ(s2.size(), 2);
	EXPECT_TRUE(s2.contains(1));
	EXPECT_TRUE(s2.contains(2));
}

TEST(HashSetTest, IsFull) {
	HashSet<int> s(2);
	EXPECT_FALSE(s.isFull());
	s.add(1);
	EXPECT_FALSE(s.isFull());
	s.add(2);
	EXPECT_TRUE(s.isFull());
	s.add(3); // resizes
	EXPECT_FALSE(s.isFull());
}

TEST(HashSetTest, RemoveAndReadd) {
	HashSet<int> s(3);
	s.add(1);
	s.add(2);
	s.add(3); // may cause probing
	s.remove(2);
	EXPECT_FALSE(s.contains(2));
	s.add(4);
	EXPECT_TRUE(s.contains(4));
	EXPECT_EQ(s.size(), 3);
}

TEST(HashSetTest, PresentAndKeyAtIndex) {
	HashSet<int> s(5);
	s.add(10);
	// Since hash, index unknown, but can loop
	int count = 0;
	for (int i = 0; i < s.getCapacity(); ++i) {
		if (s.presentAtIndex(i)) {
			++count;
			EXPECT_EQ(s.keyAtIndex(i), 10);
		}
	}
	EXPECT_EQ(count, 1);
}

TEST(HashSetTest, ContainerInterface) {
    HashSet<int> s(5);
    s.add(10);
    s.add(20);
    s.add(30);

    Container<int, int&, HashSetIterator<int>, HashSetConstIterator<int>>* container = &s;
    EXPECT_EQ(container->size(), 3);

    // Test iteration
    int count = 0;
    bool found10 = false, found20 = false, found30 = false;
    for (auto it = container->begin(); it != container->end(); ++it) {
        count++;
        if (*it == 10) found10 = true;
        else if (*it == 20) found20 = true;
        else if (*it == 30) found30 = true;
    }
    EXPECT_EQ(count, 3);
    EXPECT_TRUE(found10 && found20 && found30);

    // Test find
    EXPECT_GE(container->find(20), 0);
}

