/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-02-24
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
#include <cstdio>
#include <string>
#include <unistd.h>
#include "fcntl.h"
#include "fs/BTree.h"
#include "ds/ArrayList.h"
#include "ds/HashSet.h"
#include "universaltime.h"


static std::string makeBTreeTempFile(const char* base) {
	char buf[256];
	snprintf(buf, sizeof(buf), "%s_%llu.tmp", base, (unsigned long long)millis_since_epoch());
	return std::string(buf);
}


template <class T, int N>
class RAMBTree: public BTreeBase<T, N> {
public:
	RAMBTree(int (*compare)(const T&, const T&))
		: BTreeBase<T, N>(compare) {
		nodes.add({{0, 0, 0, -1, 0, {}}, {}});
	}

protected:

	btree_node_t<T, N> getNode(uint64_t offset) const override {
		return nodes.get((int)offset);
	}

	btree_node_t<T, N> getRootNode() const override {
		return getNode(0);
	}

	uint64_t getRootOffset() const override {
		return 0;
	}

	uint64_t addNode(const btree_node_t<T, N>& node) override {
		if (node.header.nElements > N)
			ADD_FAILURE();
		else if (node.header.nElements < N / 2)
			ADD_FAILURE();
		else if (node.header.indexInParent > getNode(node.header.parent).header.nElements + 1)
			ADD_FAILURE();
		else if (node.header.nChildren != 0 && node.header.nChildren != node.header.nElements + 1)
			ADD_FAILURE();
		else {
			int lastChildOffset = -1;
			for (int i = 0; i < node.header.nChildren; i++) {
				if ((int)node.header.childOffsets[i] >= nodes.size())
					ADD_FAILURE();
				if ((int)node.header.childOffsets[i] == lastChildOffset)
					abort();
				lastChildOffset = (int)node.header.childOffsets[i];
			}
		}
		uint64_t i = nodes.size();
		EXPECT_NE(node.header.indexInParent, -1);
		nodes.add(node);
		return i;
	}

	void overwriteNode(uint64_t offset, const btree_node_t<T, N>& node) override {
		EXPECT_LE(node.header.nElements, N);
		if (offset != getRootOffset()) {
			EXPECT_GE(node.header.nElements, N / 2);
			EXPECT_GE(node.header.indexInParent, 0);
			EXPECT_LE(node.header.indexInParent, N);
		} else {
			EXPECT_EQ(node.header.indexInParent, -1);
		}
		if (node.header.nChildren != 0) {
			EXPECT_EQ(node.header.nChildren, node.header.nElements + 1);
		}

		int lastChildOffset = -1;
		for (int i = 0; i < node.header.nChildren; i++) {
			EXPECT_LT((int)node.header.childOffsets[i], nodes.size());
			if (node.header.childOffsets[i] == offset)
				ADD_FAILURE();
			else if (getNode(node.header.childOffsets[i]).header.parent != offset)
				ADD_FAILURE();
			else if (getNode(node.header.childOffsets[i]).header.indexInParent != i)
				ADD_FAILURE();
			if ((int)node.header.childOffsets[i] == lastChildOffset)
				abort();
			lastChildOffset = (int)node.header.childOffsets[i];

		}
		nodes.set((int)offset, node);
	}

	void overwriteNodeHeader(uint64_t offset, const btree_node_t<T, N>& node) override {
		overwriteNode(offset, node);
	}

	ArrayList<btree_node_t<T, N>> nodes;

public:
	void collectInorder(ArrayList<T>& out) const {
		collectInorderFrom(0, out);
	}

	void assertHealthy(const char* where) const {
		assertHealthyNode(0, true, where);
	}

	int rootChildCount() const {
		return nodes.get(0).header.nChildren;
	}

	int rootElementCount() const {
		return nodes.get(0).header.nElements;
	}

	void collectInternalKeys(ArrayList<T>& out) const {
		for (int i = 0; i < nodes.size(); i++) {
			const btree_node_t<T, N>& node = nodes.get(i);
			if (node.header.nChildren == 0)
				continue;
			for (int e = 0; e < node.header.nElements; e++)
				out.add(node.elements[e]);
		}
	}

private:
	void collectInorderFrom(uint64_t offset, ArrayList<T>& out) const {
		const btree_node_t<T, N>& node = nodes.get((int)offset);
		if (node.header.nChildren == 0) {
			for (int i = 0; i < node.header.nElements; i++)
				out.add(node.elements[i]);
			return;
		}
		for (int i = 0; i < node.header.nElements; i++) {
			collectInorderFrom(node.header.childOffsets[i], out);
			out.add(node.elements[i]);
		}
		collectInorderFrom(node.header.childOffsets[node.header.nElements], out);
	}

	T subtreeMin(uint64_t offset) const {
		const btree_node_t<T, N>& node = nodes.get((int)offset);
		if (node.header.nChildren == 0)
			return node.elements[0];
		return subtreeMin(node.header.childOffsets[0]);
	}

	T subtreeMax(uint64_t offset) const {
		const btree_node_t<T, N>& node = nodes.get((int)offset);
		if (node.header.nChildren == 0)
			return node.elements[node.header.nElements - 1];
		return subtreeMax(node.header.childOffsets[node.header.nChildren - 1]);
	}

	void assertHealthyNode(uint64_t offset, bool isRoot, const char* where) const {
		ASSERT_LT((int)offset, nodes.size()) << where << " offset " << offset;
		const btree_node_t<T, N>& node = nodes.get((int)offset);
		EXPECT_GE(node.header.nElements, 0) << where << " off=" << offset;
		EXPECT_LE(node.header.nElements, N) << where << " off=" << offset;
		if (!isRoot) {
			EXPECT_GE(node.header.nElements, N / 2) << where << " underfull off=" << offset;
		} else {
			EXPECT_FALSE(node.header.nElements == 0 && node.header.nChildren == 1)
				<< where << " root has a single child and no keys";
		}
		if (node.header.nChildren != 0) {
			EXPECT_EQ(node.header.nChildren, node.header.nElements + 1) << where << " off=" << offset;
			EXPECT_GT(node.header.nElements, 0) << where << " internal empty off=" << offset;
		}
		for (int i = 1; i < node.header.nElements; i++) {
			EXPECT_LT(this->compare(node.elements[i - 1], node.elements[i]), 0)
				<< where << " unsorted off=" << offset << " i=" << i;
		}
		if (node.header.nChildren == 0)
			return;
		for (int i = 0; i < node.header.nChildren; i++) {
			uint64_t childOff = node.header.childOffsets[i];
			ASSERT_LT((int)childOff, nodes.size()) << where << " child off=" << childOff;
			EXPECT_NE(childOff, offset) << where << " self-child";
			const btree_node_t<T, N>& child = nodes.get((int)childOff);
			EXPECT_EQ(child.header.parent, offset) << where << " parent link child=" << childOff;
			EXPECT_EQ(child.header.indexInParent, i) << where << " indexInParent child=" << childOff;
			if (child.header.nElements > 0) {
				if (i > 0) {
					EXPECT_LT(this->compare(node.elements[i - 1], subtreeMin(childOff)), 0)
						<< where << " sep vs child min i=" << i;
				}
				if (i < node.header.nElements) {
					EXPECT_LT(this->compare(subtreeMax(childOff), node.elements[i]), 0)
						<< where << " child max vs sep i=" << i;
				}
			}
			assertHealthyNode(childOff, false, where);
		}
	}
};


static int compareInts(const int& a, const int& b) {
	return a - b;
}


static void shuffleInts(ArrayList<int>& keys, uint32_t seed) {
	for (int i = keys.size() - 1; i > 0; i--) {
		seed = seed * 1664525u + 1013904223u;
		int j = (int)(seed % (uint32_t)(i + 1));
		int tmp = keys.get(i);
		keys.set(i, keys.get(j));
		keys.set(j, tmp);
	}
}


template<int N>
static void expectContents(RAMBTree<int, N>& tree, const HashSet<int>& expected, const char* where) {
	tree.assertHealthy(where);

	for (int key : expected) {
		int val = key;
		ASSERT_TRUE(tree.find(val)) << where << " missing " << key;
		EXPECT_EQ(val, key) << where;
	}

	ArrayList<int> inorder;
	tree.collectInorder(inorder);
	ASSERT_EQ(inorder.size(), expected.size()) << where << " inorder size";
	for (int i = 0; i < inorder.size(); i++) {
		EXPECT_TRUE(expected.contains(inorder.get(i))) << where << " extra " << inorder.get(i);
		if (i > 0) {
			EXPECT_LT(inorder.get(i - 1), inorder.get(i)) << where << " inorder not sorted";
		}
	}

	if (expected.size() == 0) {
		int q = 0;
		EXPECT_FALSE(tree.findNext(q)) << where << " empty findNext";
		return;
	}

	int first = inorder.get(0);
	int probe = first;
	ASSERT_TRUE(tree.findNext(probe)) << where << " first findNext";
	EXPECT_EQ(probe, first) << where;

	for (int i = 0; i < inorder.size() - 1; i++) {
		int gap = inorder.get(i) + 1;
		if (gap < inorder.get(i + 1)) {
			int got = gap;
			ASSERT_TRUE(tree.findNext(got)) << where << " gap " << gap;
			EXPECT_EQ(got, inorder.get(i + 1)) << where << " gap " << gap;
		}
	}

	int after = inorder.get(inorder.size() - 1) + 1;
	int miss = after;
	EXPECT_FALSE(tree.findNext(miss)) << where << " past end";
	EXPECT_EQ(miss, after) << where;
	EXPECT_FALSE(tree.find(miss)) << where << " past end find";
}


class BTreeBaseTest : public ::testing::Test {
public:
	BTreeBaseTest() : tree(compareInts) {}
protected:
	RAMBTree<int, 63> tree;
};


// Basic insertion and find test
TEST_F(BTreeBaseTest, BasicInsertAndFind) {
	tree.insert(5);
	tree.insert(10);
	tree.insert(3);

	int val;
	EXPECT_TRUE(tree.find(val = 5));
	EXPECT_EQ(5, val);

	EXPECT_TRUE(tree.find(val = 10));
	EXPECT_EQ(10, val);

	EXPECT_TRUE(tree.find(val = 3));
	EXPECT_EQ(3, val);

	// Test not found case
	int not_found = -1;
	EXPECT_FALSE(tree.find(not_found));
	EXPECT_EQ(-1, not_found);
}

// Test with exactly N+1 elements (should fill the root node)
TEST_F(BTreeBaseTest, FillRootNode) {
	const int num_elements = 63 + 1; // Insert one more than can fit
	for (int i = 0; i < num_elements; ++i) {
		int h = (i * 973) % 71;
		tree.insert(h);
	}

	// Verify all elements are present
	for (int i = 0; i < num_elements; ++i) {
		int val = (i * 973) % 71;
		EXPECT_TRUE(tree.find(val));
		EXPECT_EQ((i * 973) % 71, val);
	}
}

// Test duplicate insertions
TEST_F(BTreeBaseTest, DuplicateInsertions) {
	const int value = 42;
	for (int i = 0; i < 10; ++i) {
		tree.insert(value);
	}

	// Verify we can find the value
	int val = value;
	EXPECT_TRUE(tree.find(val));
	EXPECT_EQ(value, val);

	// Verify no corruption occurred
	for (int i = 0; i < 10; ++i) {
		tree.insert(i);
		int test_val = i;
		EXPECT_TRUE(tree.find(test_val));
		EXPECT_EQ(i, test_val);
	}
}

// Stress test with many elements
TEST_F(BTreeBaseTest, StressTest) {
	uint64_t startTime = millis_since_epoch();

	const int num_elements = 40000;
	for (int i = 0; i < num_elements; ++i) {
		int h = (int)(((uint32_t)i * 27644437 + 87) ^ 0xE7D9541) % 1299827;
		tree.insert(h);
	}

	uint64_t insertionTime = millis_since_epoch() - startTime;
	startTime = millis_since_epoch();

	// Verify all elements are present
	for (int i = 0; i < num_elements; ++i) {
		int val = (int)(((uint32_t)i * 27644437 + 87) ^ 0xE7D9541) % 1299827;
		EXPECT_TRUE(tree.find(val));
		EXPECT_EQ((int)(((uint32_t)i * 27644437 + 87) ^ 0xE7D9541) % 1299827, val);
	}

	uint64_t retrieveTime = millis_since_epoch() - startTime;

	// Test some not found cases
	int not_found = -1;
	EXPECT_FALSE(tree.find(not_found));
	EXPECT_FALSE(tree.find(not_found = 1300000));

	printf("Insertion time for %i elements: %lums (%.3fus per element)\n", num_elements, insertionTime, 1000.0 * (double) insertionTime / num_elements);
	printf("Retrieval time for %i elements: %lums (%.3fus per element)\n", num_elements, retrieveTime, 1000.0 * (double) retrieveTime / num_elements);
}

// Test node splitting and balancing
TEST_F(BTreeBaseTest, SplitAndBalance) {
	const int num_elements = 100; // Number of elements to insert
	for (int i = 0; i < num_elements; ++i) {
		tree.insert(i);
	}

	// Verify all elements are present after multiple splits
	for (int i = 0; i < num_elements; ++i) {
		int val = i;
		EXPECT_TRUE(tree.find(val));
		EXPECT_EQ(i, val);
	}
}

// Test edge case with minimum node size
TEST_F(BTreeBaseTest, MinNodeSize) {
	const int min_elements = 63 + 2; // Insert one more than can fit in a single node
	for(int i = 0; i < min_elements; ++i){
		tree.insert(i);
	}

	// Verify all elements are present after splitting
	for(int i = 0; i < min_elements; ++i){
		int val = -1;
		EXPECT_TRUE(tree.find(val = i));
		EXPECT_EQ(i, val);
	}
}

// Stress test with many elements
// Does not work in CI for some reason
/*TEST_F(BTreeBaseTest, DiskStressTest) {
	uint64_t startTime = millis_since_epoch();

	FdHandle file = FdHandle::open("/tmp/BTreeDiskStressTest.bin", O_RDWR | O_CREAT, 0660);
	BTree<int, 7> tree(file, 0, compareInts);
	tree.initialize();
	file.flush();

	const int num_elements = 4000;
	for (int i = 0; i < num_elements; ++i) {
		int h = ((i * 27644437 + 87) ^ 0xE7D9541) % 1299827;
		tree.insert(h);
	}

	uint64_t insertionTime = millis_since_epoch() - startTime;
	startTime = millis_since_epoch();

	// Verify all elements are present
	for (int i = 0; i < num_elements; ++i) {
		int val = ((i * 27644437 + 87) ^ 0xE7D9541) % 1299827;
		EXPECT_TRUE(tree.find(val));
		EXPECT_EQ(((i * 27644437 + 87) ^ 0xE7D9541) % 1299827, val);
	}

	uint64_t retrieveTime = millis_since_epoch() - startTime;

	// Test some not found cases
	int not_found = -1;
	EXPECT_FALSE(tree.find(not_found));
	EXPECT_FALSE(tree.find(not_found = 1300000));

	printf("Insertion time for %i elements: %lums (%.3fus per element)\n", num_elements, insertionTime, 1000.0 * (double) insertionTime / num_elements);
	printf("Retrieval time for %i elements: %lums (%.3fus per element)\n", num_elements, retrieveTime, 1000.0 * (double) retrieveTime / num_elements);
}*/

TEST_F(BTreeBaseTest, RemoveMissingFromEmpty) {
	int val = 7;
	EXPECT_FALSE(tree.remove(val));
	EXPECT_EQ(val, 7);
	tree.assertHealthy("empty");
	EXPECT_FALSE(tree.find(val));
}


TEST_F(BTreeBaseTest, RemoveFromRootLeaf) {
	tree.insert(5);
	tree.insert(10);
	tree.insert(3);

	int val = 10;
	EXPECT_TRUE(tree.remove(val));
	EXPECT_FALSE(tree.find(val));

	HashSet<int> expected = HashSet<int>(4);
	expected.add(3);
	expected.add(5);
	expectContents(tree, expected, "after remove 10");

	val = 10;
	EXPECT_FALSE(tree.remove(val));

	val = 5;
	EXPECT_TRUE(tree.remove(val));
	expected.remove(5);
	expectContents(tree, expected, "after remove 5");

	val = 3;
	EXPECT_TRUE(tree.remove(val));
	expected.remove(3);
	expectContents(tree, expected, "empty after last");

	tree.insert(1);
	expected.add(1);
	expectContents(tree, expected, "insert after empty");
}


template<int N>
static void insertRange(RAMBTree<int, N>& tree, HashSet<int>& expected, int lo, int hi) {
	for (int i = lo; i < hi; i++) {
		EXPECT_FALSE(tree.insert(i)) << i;
		EXPECT_FALSE(expected.add(i)) << i;
	}
}


TEST(BTreeRemove, SequentialForwardN3) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(64);
	insertRange(t, expected, 0, 40);
	expectContents(t, expected, "inserted");

	for (int i = 0; i < 40; i++) {
		int val = i;
		ASSERT_TRUE(t.remove(val)) << i;
		expected.remove(i);
		expectContents(t, expected, "forward");
		if (HasFailure())
			return;
	}
}


TEST(BTreeRemove, SequentialReverseN3) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(64);
	insertRange(t, expected, 0, 40);
	for (int i = 39; i >= 0; i--) {
		int val = i;
		ASSERT_TRUE(t.remove(val)) << i;
		expected.remove(i);
		expectContents(t, expected, "reverse");
		if (HasFailure())
			return;
	}
}


TEST(BTreeRemove, EveryOtherThenRestN3) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(64);
	insertRange(t, expected, 0, 50);
	for (int i = 0; i < 50; i += 2) {
		int val = i;
		ASSERT_TRUE(t.remove(val)) << i;
		expected.remove(i);
		expectContents(t, expected, "evens");
		if (HasFailure())
			return;
	}
	for (int i = 1; i < 50; i += 2) {
		int val = i;
		ASSERT_TRUE(t.remove(val)) << i;
		expected.remove(i);
		expectContents(t, expected, "odds");
		if (HasFailure())
			return;
	}
}


TEST(BTreeRemove, ShuffledN3) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(128);
	const int n = 80;
	insertRange(t, expected, 0, n);

	ArrayList<int> order;
	for (int i = 0; i < n; i++)
		order.add(i);
	shuffleInts(order, 0xBEEF1234u);

	for (int i = 0; i < order.size(); i++) {
		int key = order.get(i);
		int val = key;
		ASSERT_TRUE(t.remove(val)) << key;
		expected.remove(key);
		expectContents(t, expected, "shuffled");
		if (HasFailure())
			return;
	}
}


TEST(BTreeRemove, ShuffledN63) {
	RAMBTree<int, 63> t(compareInts);
	HashSet<int> expected = HashSet<int>(256);
	const int n = 200;
	insertRange(t, expected, 0, n);

	ArrayList<int> order;
	for (int i = 0; i < n; i++)
		order.add(i);
	shuffleInts(order, 0xA5A5A5A5u);

	for (int i = 0; i < order.size(); i++) {
		int key = order.get(i);
		int val = key;
		ASSERT_TRUE(t.remove(val)) << key;
		expected.remove(key);
		expectContents(t, expected, "shuffled63");
		if (HasFailure())
			return;
	}
}


TEST(BTreeRemove, InternalKeysN3) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(64);
	insertRange(t, expected, 0, 30);

	ArrayList<int> internals;
	t.collectInternalKeys(internals);
	ASSERT_GT(internals.size(), 0);

	for (int i = 0; i < internals.size(); i++) {
		int key = internals.get(i);
		if (!expected.contains(key))
			continue;
		int val = key;
		ASSERT_TRUE(t.remove(val)) << key;
		expected.remove(key);
		expectContents(t, expected, "internal");
		if (HasFailure())
			return;
	}
}


TEST(BTreeRemove, RootShrinksToLeafThenEmpty) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(64);
	insertRange(t, expected, 0, 20);
	ASSERT_GT(t.rootChildCount(), 0);

	for (int i = 0; i < 20; i++) {
		int val = i;
		ASSERT_TRUE(t.remove(val)) << i;
		expected.remove(i);
		expectContents(t, expected, "shrink");
		if (HasFailure())
			return;
	}
	EXPECT_EQ(t.rootChildCount(), 0);
	EXPECT_EQ(t.rootElementCount(), 0);
}


TEST(BTreeRemove, InterleavedInsertDelete) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(256);
	uint32_t seed = 0xC0FFEEu;
	const int universe = 120;
	for (int step = 0; step < 800; step++) {
		seed = seed * 1664525u + 1013904223u;
		int key = (int)(seed % (uint32_t)universe);
		int val = key;
		if ((seed & 1u) == 0) {
			bool already = t.insert(key);
			EXPECT_EQ(already, expected.contains(key)) << "insert " << key;
			expected.add(key);
		} else {
			bool present = expected.contains(key);
			EXPECT_EQ(t.remove(val), present) << "remove " << key;
			if (present)
				expected.remove(key);
		}
		if (step % 7 == 0 || step == 799) {
			expectContents(t, expected, "mix");
			if (HasFailure())
				return;
		}
	}
	expectContents(t, expected, "mix end");
}


TEST(BTreeRemove, ReinsertAfterDelete) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(64);
	insertRange(t, expected, 0, 25);
	for (int i = 0; i < 25; i += 3) {
		int val = i;
		ASSERT_TRUE(t.remove(val));
		expected.remove(i);
	}
	expectContents(t, expected, "before reinsert");
	for (int i = 0; i < 25; i += 3) {
		EXPECT_FALSE(t.insert(i));
		expected.add(i);
	}
	expectContents(t, expected, "after reinsert");
}


struct KvEntry {
	int key;
	int value;

	static int compare(const KvEntry& a, const KvEntry& b) {
		return a.key - b.key;
	}
};


TEST(BTreeRemove, KeyValuePayload) {
	RAMBTree<KvEntry, 3> t(KvEntry::compare);
	for (int i = 0; i < 40; i++)
		EXPECT_FALSE(t.insert(KvEntry{i, i * 10}));

	KvEntry q{17, -1};
	ASSERT_TRUE(t.find(q));
	EXPECT_EQ(q.value, 170);

	q = KvEntry{17, 0};
	ASSERT_TRUE(t.remove(q));
	q = KvEntry{17, 0};
	EXPECT_FALSE(t.find(q));
	EXPECT_FALSE(t.remove(q));

	t.assertHealthy("after kv remove 17");

	q = KvEntry{16, 0};
	ASSERT_TRUE(t.find(q));
	EXPECT_EQ(q.value, 160);
	q = KvEntry{18, 0};
	ASSERT_TRUE(t.find(q));
	EXPECT_EQ(q.value, 180);

	for (int i = 0; i < 40; i++) {
		if (i == 17)
			continue;
		q = KvEntry{i, 0};
		ASSERT_TRUE(t.find(q)) << i;
		EXPECT_EQ(q.value, i * 10) << i;
	}
}


TEST(BTreeRemove, RandomMixStressN3SecondSeed) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(2048);
	uint32_t seed = 0x0D15EA5Eu;
	const int universe = 300;
	for (int step = 0; step < 2000; step++) {
		seed = seed * 1664525u + 1013904223u;
		int key = (int)(seed % (uint32_t)universe);
		int val = key;
		if ((seed >> 16) & 1u) {
			bool already = t.insert(key);
			EXPECT_EQ(already, expected.contains(key)) << "insert " << key;
			expected.add(key);
		} else {
			bool present = expected.contains(key);
			EXPECT_EQ(t.remove(val), present) << "remove " << key;
			if (present)
				expected.remove(key);
		}
		if (step % 11 == 0 || step == 1999) {
			expectContents(t, expected, "mix2");
			if (HasFailure())
				return;
		}
	}
	expectContents(t, expected, "mix2 end");
}


TEST(BTreeRemove, RandomMixStressN3) {
	RAMBTree<int, 3> t(compareInts);
	HashSet<int> expected = HashSet<int>(1024);
	const int n = 400;
	insertRange(t, expected, 0, n);

	ArrayList<int> order;
	for (int i = 0; i < n; i++)
		order.add(i);
	shuffleInts(order, 0x13579BDFu);

	for (int i = 0; i < n / 2; i++) {
		int key = order.get(i);
		int val = key;
		ASSERT_TRUE(t.remove(val)) << key;
		expected.remove(key);
	}
	expectContents(t, expected, "half gone");

	for (int i = 0; i < n / 2; i++) {
		int key = order.get(i);
		EXPECT_FALSE(t.insert(key + n));
		expected.add(key + n);
	}
	expectContents(t, expected, "refilled high");

	shuffleInts(order, 0x2468ACE0u);
	for (int i = 0; i < order.size(); i++) {
		int key = order.get(i);
		int val = key;
		if (expected.contains(key)) {
			ASSERT_TRUE(t.remove(val)) << key;
			expected.remove(key);
		} else {
			EXPECT_FALSE(t.remove(val)) << key;
		}
	}
	expectContents(t, expected, "after second pass");
}


TEST_F(BTreeBaseTest, FindNext) {
	const int num_elements = 100;
	for (int i = 0; i < num_elements; ++i) {
		tree.insert(i * 2);
	}

	tree.insert(5000);

	int val = 95;
	EXPECT_TRUE(tree.findNext(val));
	EXPECT_EQ(96, val);

	val = 1000;
	EXPECT_TRUE(tree.findNext(val));
	EXPECT_EQ(5000, val);

	val = 10000;
	EXPECT_FALSE(tree.findNext(val));
	EXPECT_EQ(10000, val);
}

TEST(BTreeFindNext, FindsClosestHigherInChildNode) {
	RAMBTree<int, 3> tree(compareInts);

	// Insert enough elements to force multiple splits and a multi-level tree.
	// Using even numbers so we can probe with odd "gap" queries.
	const int numElements = 200;
	for (int i = 0; i < numElements; ++i)
		tree.insert(i * 2);

	// Sanity check: every inserted value must be retrievable via find().
	for (int i = 0; i < numElements; ++i) {
		int val = i * 2;
		ASSERT_TRUE(tree.find(val)) << "find missing for " << (i * 2);
		ASSERT_EQ(i * 2, val);
	}

	// For every odd value in range, findNext must return the next even value.
	// Under the previous buggy findNext, the descent skipped the correct child
	// and returned an unrelated greater value (or returned false), making this
	// test fail.
	for (int q = 1; q < numElements * 2 - 1; q += 2) {
		int val = q;
		EXPECT_TRUE(tree.findNext(val)) << "query=" << q;
		EXPECT_EQ(q + 1, val) << "query=" << q;
	}

	// Exact-match probes must still work.
	for (int q = 0; q < numElements * 2; q += 2) {
		int val = q;
		EXPECT_TRUE(tree.findNext(val)) << "query=" << q;
		EXPECT_EQ(q, val) << "query=" << q;
	}

	// A value strictly larger than every element should not be found.
	int val = numElements * 2 + 10;
	EXPECT_FALSE(tree.findNext(val));
	EXPECT_EQ(numElements * 2 + 10, val);
}


// Regression test: findNext with a sparse outlier in a deep tree.  Searching
// in the gap between the densely packed range and the outlier must return
// the outlier (which lives in a different subtree than the descent path of
// the buggy implementation).
TEST(BTreeFindNext, FindsOutlierAcrossSubtrees) {
	RAMBTree<int, 3> tree(compareInts);

	for (int i = 0; i < 100; ++i)
		tree.insert(i * 2);
	tree.insert(100000);

	int val = 500;
	EXPECT_TRUE(tree.findNext(val));
	EXPECT_EQ(100000, val);

	val = 99999;
	EXPECT_TRUE(tree.findNext(val));
	EXPECT_EQ(100000, val);

	val = 100001;
	EXPECT_FALSE(tree.findNext(val));
}


TEST(BTreeDisk, InitializeInsertFindAndReopen) {
	std::string path = makeBTreeTempFile("btree_disk_ok");
	unlink(path.c_str());

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR | O_CREAT, 0660);
		ASSERT_TRUE((bool)file);
		BTree<int, 3> tree(file, 0, compareInts);
		for (int i = 0; i < 50; ++i)
			EXPECT_FALSE(tree.insert(i));
		for (int i = 0; i < 50; ++i) {
			int v = i;
			ASSERT_TRUE(tree.find(v));
			EXPECT_EQ(v, i);
		}
		int miss = 1000;
		EXPECT_FALSE(tree.find(miss));
	}

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
		ASSERT_TRUE((bool)file);
		BTree<int, 3> tree(file, 0, compareInts);
		for (int i = 0; i < 50; ++i) {
			int v = i;
			ASSERT_TRUE(tree.find(v)) << i;
			EXPECT_EQ(v, i);
		}
	}

	unlink(path.c_str());
}


TEST(BTreeDisk, RemoveAndReopen) {
	std::string path = makeBTreeTempFile("btree_disk_remove");
	unlink(path.c_str());

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR | O_CREAT, 0660);
		ASSERT_TRUE((bool)file);
		BTree<int, 3> disk(file, 0, compareInts);
		for (int i = 0; i < 80; i++)
			EXPECT_FALSE(disk.insert(i));
		for (int i = 0; i < 80; i += 2) {
			int val = i;
			ASSERT_TRUE(disk.remove(val)) << i;
		}
		for (int i = 1; i < 80; i += 2) {
			int val = i;
			ASSERT_TRUE(disk.find(val)) << i;
			EXPECT_EQ(val, i);
		}
		int gone = 4;
		EXPECT_FALSE(disk.find(gone));
	}

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
		ASSERT_TRUE((bool)file);
		BTree<int, 3> disk(file, 0, compareInts);
		for (int i = 1; i < 80; i += 2) {
			int val = i;
			ASSERT_TRUE(disk.find(val)) << i;
			EXPECT_EQ(val, i);
			ASSERT_TRUE(disk.remove(val)) << i;
		}
		int any = 1;
		EXPECT_FALSE(disk.find(any));
		EXPECT_FALSE(disk.findNext(any));
		EXPECT_FALSE(disk.insert(99));
		int v = 99;
		ASSERT_TRUE(disk.find(v));
	}

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
		BTree<int, 3> disk(file, 0, compareInts);
		int v = 99;
		ASSERT_TRUE(disk.find(v));
		EXPECT_EQ(v, 99);
		int miss = 1;
		EXPECT_FALSE(disk.find(miss));
	}

	unlink(path.c_str());
}


namespace {

template <int N>
void writeCraftedBTree(const char* path, const btree_node_t<int, N>* nodes, int count) {
	unlink(path);
	FdHandle h = FdHandle::open(path, O_RDWR | O_CREAT | O_TRUNC, 0660);
	ASSERT_TRUE((bool)h);
	for (int i = 0; i < count; ++i) {
		off_t off = (off_t)i * (off_t)sizeof(nodes[i]);
		ASSERT_EQ(h.pwrite(nodes[i], off), (ssize_t)sizeof(nodes[i]));
	}
	h.flush();
}

}


TEST(BTreeSecurity, RejectsInvalidMagic) {
	std::string path = makeBTreeTempFile("btree_bad_magic");
	auto node = makeBTreeNode<int, 3>();
	node.header.magic = 0xDEAD;
	writeCraftedBTree<3>(path.c_str(), &node, 1);

	FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
	EXPECT_THROW((BTree<int, 3>(file, 0, compareInts)), BTreeCorruptError);
	unlink(path.c_str());
}


TEST(BTreeSecurity, AcceptsLegacyUnstampedFormat) {
	// Nodes written before magic/version existed have 0 in that word.
	// They must still open, be searchable, and accept new inserts.
	std::string path = makeBTreeTempFile("btree_legacy");
	auto node = makeBTreeNode<int, 3>();
	node.header.magic = 0;
	node.header.version = 0;
	node.header.nElements = 2;
	node.elements[0] = 10;
	node.elements[1] = 20;
	writeCraftedBTree<3>(path.c_str(), &node, 1);

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
		BTree<int, 3> tree(file, 0, compareInts);
		int v = 10;
		ASSERT_TRUE(tree.find(v));
		EXPECT_EQ(v, 10);
		v = 20;
		ASSERT_TRUE(tree.find(v));
		EXPECT_FALSE(tree.insert(15));
		v = 15;
		ASSERT_TRUE(tree.find(v));
	}

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
		BTree<int, 3> tree(file, 0, compareInts);
		int v = 15;
		ASSERT_TRUE(tree.find(v));
		EXPECT_EQ(v, 15);
	}

	// Count checks still apply to the old format.
	node.header.nElements = 1000;
	writeCraftedBTree<3>(path.c_str(), &node, 1);
	FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
	EXPECT_THROW((BTree<int, 3>(file, 0, compareInts)), BTreeCorruptError);
	unlink(path.c_str());
}


TEST(BTreeSecurity, RejectsNElementsOutOfRange) {
	std::string path = makeBTreeTempFile("btree_bad_nelems");
	auto node = makeBTreeNode<int, 3>();
	node.header.nElements = 1000;
	writeCraftedBTree<3>(path.c_str(), &node, 1);

	FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
	EXPECT_THROW((BTree<int, 3>(file, 0, compareInts)), BTreeCorruptError);
	unlink(path.c_str());
}


TEST(BTreeSecurity, RejectsNChildrenOutOfRange) {
	std::string path = makeBTreeTempFile("btree_bad_nchildren");
	auto node = makeBTreeNode<int, 3>();
	node.header.nChildren = 1000;
	writeCraftedBTree<3>(path.c_str(), &node, 1);

	FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
	EXPECT_THROW((BTree<int, 3>(file, 0, compareInts)), BTreeCorruptError);
	unlink(path.c_str());
}


TEST(BTreeSecurity, RejectsShortFile) {
	std::string path = makeBTreeTempFile("btree_short");
	unlink(path.c_str());
	{
		FdHandle h = FdHandle::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0660);
		ASSERT_TRUE((bool)h);
		uint32_t junk = 0x41414141;
		h.pwrite(junk, 0);
		h.flush();
	}

	FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
	EXPECT_THROW((BTree<int, 3>(file, 0, compareInts)), BTreeCorruptError);
	unlink(path.c_str());
}


TEST(BTreeSecurity, RejectsHugeChildOffset) {
	std::string path = makeBTreeTempFile("btree_huge_child");
	auto node = makeBTreeNode<int, 3>(0, 2, 1, -1);
	node.elements[0] = 50;
	node.header.childOffsets[0] = (uint64_t)1 << 60;
	node.header.childOffsets[1] = (uint64_t)1 << 60;
	writeCraftedBTree<3>(path.c_str(), &node, 1);

	FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
	BTree<int, 3> tree(file, 0, compareInts);
	int q = 10;
	EXPECT_THROW(tree.find(q), BTreeCorruptError);
	q = 10;
	EXPECT_THROW(tree.findNext(q), BTreeCorruptError);
	unlink(path.c_str());
}


TEST(BTreeSecurity, InsertDoesNotWriteCraftedParent) {
	// Full leaf whose parent offset points at a decoy region with an
	// unrecognized stamp.  insert() must throw instead of pwrite-ing there.
	// (A zero-filled region would be the legacy unstamped format and is valid.)
	std::string path = makeBTreeTempFile("btree_bad_parent");
	using Node = btree_node_t<int, 3>;
	const off_t decoyOff = (off_t)sizeof(Node);

	Node root = makeBTreeNode<int, 3>(0, 2, 1, -1);
	root.elements[0] = 50;
	root.header.childOffsets[0] = (uint64_t)(2 * sizeof(Node));
	root.header.childOffsets[1] = (uint64_t)(3 * sizeof(Node));

	Node decoy = makeBTreeNode<int, 3>();
	decoy.header.magic = 0xDEAD;

	Node left = makeBTreeNode<int, 3>((uint64_t)decoyOff, 0, 3, 0);
	left.elements[0] = 10;
	left.elements[1] = 20;
	left.elements[2] = 30;

	Node right = makeBTreeNode<int, 3>(0, 0, 1, 1);
	right.elements[0] = 60;

	unlink(path.c_str());
	{
		FdHandle h = FdHandle::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0660);
		ASSERT_TRUE((bool)h);
		ASSERT_EQ(h.pwrite(root, 0), (ssize_t)sizeof(root));
		ASSERT_EQ(h.pwrite(decoy, decoyOff), (ssize_t)sizeof(decoy));
		ASSERT_EQ(h.pwrite(left, (off_t)(2 * sizeof(Node))), (ssize_t)sizeof(left));
		ASSERT_EQ(h.pwrite(right, (off_t)(3 * sizeof(Node))), (ssize_t)sizeof(right));
		h.flush();
	}

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
		BTree<int, 3> tree(file, 0, compareInts);
		EXPECT_THROW(tree.insert(15), BTreeCorruptError);
	}

	{
		FdHandle file = FdHandle::open(path.c_str(), O_RDWR, 0660);
		Node got{};
		ASSERT_EQ(file.pread(got, decoyOff), (ssize_t)sizeof(got));
		EXPECT_EQ(got.header.magic, (uint16_t)0xDEAD);
	}

	unlink(path.c_str());
}
/*
class BTreeIteratorTest : public ::testing::Test {
public:
	BTreeIteratorTest() : tree(compareInts) {}
protected:
	RAMBTree<int, 63> tree;
};

TEST_F(BTreeIteratorTest, EmptyTree) {
	auto it = tree.begin();
	auto end = tree.end();
	EXPECT_EQ(it, end);
	EXPECT_TRUE(it == tree.end());
}

TEST_F(BTreeIteratorTest, SingleElement) {
	tree.insert(42);
	auto it = tree.begin();
	EXPECT_EQ(*it, 42);
	++it;
	EXPECT_EQ(it, tree.end());
}

TEST_F(BTreeIteratorTest, MultipleElements) {
	tree.insert(10);
	tree.insert(5);
	tree.insert(20);
	tree.insert(15);

	ArrayList<int> expected({5, 10, 15, 20});
	auto it = tree.begin();
	for (int val : expected) {
		EXPECT_EQ(*it, val);
		++it;
	}
	EXPECT_EQ(it, tree.end());
}

TEST_F(BTreeIteratorTest, LargeTreeTraversal) {
	const int num_elements = 100;
	for (int i = 0; i < num_elements; ++i) {
		tree.insert(i);
	}

	auto it = tree.begin();
	for (int i = 0; i < num_elements; ++i) {
		EXPECT_EQ(*it, i);
		++it;
	}
	EXPECT_EQ(it, tree.end());
}*/
