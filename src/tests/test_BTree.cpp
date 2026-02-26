//
// Created by nuclaer on 24.2.25.
//

#include <gtest/gtest.h>
#include "fcntl.h"
#include "fs/BTree.h"
#include "ds/ArrayList.h"
#include "universaltime.h"


template <class T, int N>
class RAMBTree: public BTreeBase<T, N> {
public:
	RAMBTree(int (*compare)(const T &, const T &))
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
		}
		EXPECT_LE(node.header.indexInParent, getNode(node.header.parent).header.nElements + 1);
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
};


static int compareInts(const int& a, const int& b) {
	return a - b;
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

	const int num_elements = 100000;
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
TEST_F(BTreeBaseTest, DiskStressTest) {
	uint64_t startTime = millis_since_epoch();

	FdHandle file = FdHandle::open("/tmp/BTreeDiskStressTest.bin", O_RDWR | O_CREAT, 0660);
	BTree<int, 7> tree(file, 0, compareInts);
	tree.initialize();
	file.flush();

	const int num_elements = 100000;
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
}

// TODO: It is known that remove does not work properly
/*TEST_F(BTreeBaseTest, BasicInsertAndRemove) {
	const int num_elements = 100;
	for (int i = 0; i < num_elements; ++i) {
		tree.insert(i);
	}

	int val;
	EXPECT_TRUE(tree.find(val = 5));
	EXPECT_EQ(5, val);

	EXPECT_TRUE(tree.find(val = 10));
	EXPECT_EQ(10, val);

	tree.remove(val);

	EXPECT_TRUE(tree.find(val = 3));
	EXPECT_EQ(3, val);

	int not_found = -1;
	EXPECT_FALSE(tree.find(not_found));
	EXPECT_EQ(-1, not_found);
}*/


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
