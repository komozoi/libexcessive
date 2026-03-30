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

    auto it = list.begin();
    EXPECT_EQ(*it, 1);
    ++it;
    EXPECT_EQ(*it, 2);
    --it;
    EXPECT_EQ(*it, 1);

    auto it_end = list.end();
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
    for (auto it = list.rbegin(); it != list.rend(); ++it) {
        EXPECT_EQ(*it, expected[count++]);
    }
    EXPECT_EQ(count, 3);
}
