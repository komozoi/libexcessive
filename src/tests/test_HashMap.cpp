/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-07-26
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


// test_hashmap.cpp

#include "ds/HashMap.h"
#include "ds/ArrayList.h"
#include <gtest/gtest.h>
#include <string>
#include <algorithm>


class DestructorCounter {
public:
	static int count;

	DestructorCounter() { }
	DestructorCounter(const DestructorCounter&) { }
	DestructorCounter(DestructorCounter&&) noexcept { }
	DestructorCounter& operator=(const DestructorCounter&) { return *this; }
	DestructorCounter& operator=(DestructorCounter&&) noexcept { return *this; }
	~DestructorCounter() {
		count++;
	}
};

int DestructorCounter::count = 0;


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
	EXPECT_EQ(map.size(), 0);
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
	EXPECT_EQ(map.size(), 3);
	EXPECT_EQ(map.get(3), "three");
}

TEST(HashMapTest, ClearMap) {
	HashMap<int, std::string> map(4);
	map.put(1, "one");
	map.put(2, "two");
	map.clear();

	EXPECT_EQ(map.size(), 0);
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

TEST(HashMapStringTest, PutAndGet) {
	HashMap<std::string, int> map(10);
	map.put("hello", 1);
	map.put("world", 2);

	EXPECT_EQ(map.get("hello"), 1);
	EXPECT_EQ(map.get("world"), 2);
	EXPECT_TRUE(map.hasKey("hello"));
	EXPECT_TRUE(map.hasKey("world"));
	EXPECT_FALSE(map.hasKey("notfound"));
}

TEST(HashMapStringTest, Overwrite) {
	HashMap<std::string, int> map(10);
	map.put("test", 100);
	EXPECT_EQ(map.get("test"), 100);
	map.put("test", 200);
	EXPECT_EQ(map.get("test"), 200);
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

TEST(HashMapTest, ContainerInterface) {
    HashMap<int, std::string> map(8);
    map.put(1, "one");
    map.put(2, "two");
    map.put(3, "three");

    Container<MapElement<int, std::string>, MapElement<int, std::string>, MapElement<int, std::string>::Iterator, MapElement<int, std::string>::ConstIterator>* container = &map;
    
    EXPECT_EQ(container->size(), 3);
    
    // Test iteration
    int count = 0;
    for (auto it = container->begin(); it != container->end(); ++it) {
        count++;
        MapElement<int, std::string> elem = *it;
        if (elem.key == 1) EXPECT_EQ(elem.value, "one");
        else if (elem.key == 2) EXPECT_EQ(elem.value, "two");
        else if (elem.key == 3) EXPECT_EQ(elem.value, "three");
        else FAIL() << "Unexpected key: " << elem.key;
    }
    EXPECT_EQ(count, 3);

    // Test getElement (indices are based on internal hash map order, but should be valid)
    MapElement<int, std::string> e0 = container->getElement(0);
    EXPECT_TRUE(e0.key == 1 || e0.key == 2 || e0.key == 3);
}

TEST(HashMapTest, ReverseIteration) {
    HashMap<int, int> map(10);
    map.put(1, 100);
    map.put(2, 200);
    map.put(3, 300);
    
    ArrayList<int> forwardKeys;
    ArrayList<int> forwardValues;
    for (auto elem : map) {
        forwardKeys.add(elem.key);
        forwardValues.add(elem.value);
    }
    std::reverse(forwardKeys.begin(), forwardKeys.end());
    std::reverse(forwardValues.begin(), forwardValues.end());
    
    int count = 0;
    for (auto it = map.rbegin(); it != map.rend(); ++it) {
        EXPECT_EQ((*it).key, forwardKeys.get(count));
        EXPECT_EQ((*it).value, forwardValues.get(count));
        count++;
    }
    EXPECT_EQ(count, 3);
}

TEST(HashMapTest, ConstReverseIteration) {
    HashMap<int, int> map(10);
    map.put(1, 100);
    map.put(2, 200);
    map.put(3, 300);
    const HashMap<int, int>& cmap = map;
    
    ArrayList<int> forwardKeys;
    ArrayList<int> forwardValues;
    for (auto elem : cmap) {
        forwardKeys.add(elem.key);
        forwardValues.add(elem.value);
    }
    std::reverse(forwardKeys.begin(), forwardKeys.end());
    std::reverse(forwardValues.begin(), forwardValues.end());
    
    int count = 0;
    for (auto it = cmap.crbegin(); it != cmap.crend(); ++it) {
        EXPECT_EQ((*it).key, forwardKeys.get(count));
        EXPECT_EQ((*it).value, forwardValues.get(count));
        count++;
    }
    EXPECT_EQ(count, 3);
}

TEST(HashMapDropTest, BasicRemoval) {
    HashMap<int, std::string> map(8);
    map.put(1, "one");
    map.put(2, "two");

    EXPECT_TRUE(map.hasKey(1));
    EXPECT_TRUE(map.drop(1));
    EXPECT_FALSE(map.hasKey(1));
    EXPECT_EQ(map.size(), 1);
    EXPECT_EQ(map.get(2), "two");
}

TEST(HashMapDropTest, ReturnValue) {
    HashMap<int, std::string> map(8);
    map.put(1, "one");

    EXPECT_TRUE(map.drop(1));
    EXPECT_FALSE(map.drop(1));
    EXPECT_FALSE(map.drop(2));
}

TEST(HashMapDropTest, KeepsOrganization) {
    // We want to force a collision to test if the chain is fixed.
    // Since HashMap uses linear probing: index = hashedKey % capacity.
    // If we have two keys that hash to the same index (or consecutive ones),
    // removing the first one should not break access to the second one.

    // For a capacity of 4, if we find two keys with same (hash % 4).
    HashMap<int, int> map(4);

    // We need to find two keys k1, k2 such that (h(k1)%4) == (h(k2)%4)
    struct GetIdx {
        uint32_t operator()(int k) {
            uint32_t hashedKey = (uint32_t)std::hash<int>{}(k);
            return hashedKey % 4;
        }
    } getIdx;

    int k1 = 0, k2 = 0;
    bool found = false;
    for (int i = 0; i < 100 && !found; ++i) {
        for (int j = i + 1; j < 100; ++j) {
            if (getIdx(i) == getIdx(j)) {
                k1 = i;
                k2 = j;
                found = true;
                break;
            }
        }
    }
    ASSERT_TRUE(found);

    map.put(k1, 100);
    map.put(k2, 200);

    // Verify they are in the same chain (one right after another or wrapped)
    int idx1 = -1;
    for(int i=0; i<4; ++i) {
        if (map.presentAtIndex(i) && map.keyAtIndex(i) == k1) idx1 = i;
    }
    int idx2 = -1;
    for(int i=0; i<4; ++i) {
        if (map.presentAtIndex(i) && map.keyAtIndex(i) == k2) idx2 = i;
    }

    // idx2 should be (idx1 + 1) % 4 if they collided.
    ASSERT_NE(idx1, -1);
    ASSERT_NE(idx2, -1);
    EXPECT_EQ(idx2, (idx1 + 1) % 4);

    // Now drop k1. k2 should still be found.
    EXPECT_TRUE(map.drop(k1));
    EXPECT_FALSE(map.hasKey(k1));
    EXPECT_TRUE(map.hasKey(k2));
    EXPECT_EQ(map.get(k2), 200);
}

TEST(HashMapDropTest, DestructorCalled) {
    DestructorCounter::count = 0;
    {
        HashMap<int, DestructorCounter> map(8);
        map.put(1, DestructorCounter());
        // One for the temp object, one for the copy into the map.
        // Actually put(K, T) takes by const ref.
        // so: 1. Temporary DestructorCounter() created.
        //     2. map.put copies it into map.
        //     3. Temporary is destroyed at end of statement.

        int initial_count = DestructorCounter::count;
        map.drop(1);
        // Should increment count by 1 (the one in the map).
        EXPECT_EQ(DestructorCounter::count, initial_count + 1);
    }
}

TEST(HashMapDropTest, AmountUsedCorrect) {
    HashMap<int, int> map(16);
    // Force some re-insertions by using keys that collide.
    // Capacity 16.
	struct GetIdx {
		uint32_t operator()(int k) {
			uint32_t hashedKey = (uint32_t)std::hash<int>{}(k);
			return hashedKey % 16;
		}
	} getIdx;

    // Find keys that collide at same index
    ArrayList<int> keys;
    int targetIdx = (int)getIdx(0);
    keys.add(0);
    for (int i = 1; keys.size() < 5; ++i) {
        if ((int)getIdx(i) == targetIdx) {
            keys.add(i);
        }
    }

    for (int i = 0; i < keys.size(); ++i) {
        map.put(keys.get(i), i);
    }

    int initialSize = map.size();
    EXPECT_EQ(initialSize, 5);

    // Drop the first one, which should trigger re-insertion of others.
    map.drop(keys.get(0));

    EXPECT_EQ(map.size(), initialSize - 1);
    for (int i = 1; i < keys.size(); ++i) {
        EXPECT_TRUE(map.hasKey(keys.get(i)));
    }
}
