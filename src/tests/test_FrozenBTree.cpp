/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-05-27
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
#include <random>
#include <algorithm>

#include "fs/FrozenBTree.h"
#include "ds/ArrayList.h"


namespace {

int compareInts(const int& a, const int& b) {
	return a - b;
}

// Builds a tree from a sorted int range [lo, hi) with the given step and
// runs find/findNext over an exhaustive set of probes.
template <int N>
void runIntRangeChecks(int lo, int hi, int step) {
	ArrayList<int> items;
	for (int v = lo; v < hi; v += step)
		items.add(v);

	ArrayList<char> buffer;
	FrozenBTree<int, N>::build(buffer, items);
	ASSERT_GE(buffer.size(), (int)sizeof(btree_node_t<int, N>));

	FrozenBTree<int, N> tree(buffer.getMemory(), compareInts);

	// Every stored value must be found.
	for (int i = 0; i < items.size(); ++i) {
		int q = items.get(i);
		ASSERT_TRUE(tree.find(q)) << "i=" << i << " q=" << q;
		EXPECT_EQ(q, items.get(i));
	}

	// findNext returns the smallest stored value >= the probe.
	for (int v = lo - 2 * step; v < hi + 2 * step; ++v) {
		int q = v;
		bool ok = tree.findNext(q);
		if (v <= lo) {
			ASSERT_TRUE(ok) << "v=" << v;
			EXPECT_EQ(q, lo);
		} else if (v >= hi - step + 1 && step == 1) {
			// Past last item.
			if (v > hi - step) {
				EXPECT_FALSE(ok) << "v=" << v;
			}
		} else {
			// Find smallest items[i] >= v by linear scan.
			int expected = -1;
			for (int i = 0; i < items.size(); ++i) {
				if (items.get(i) >= v) {
					expected = items.get(i);
					break;
				}
			}
			if (expected == -1) {
				EXPECT_FALSE(ok) << "v=" << v;
			} else {
				ASSERT_TRUE(ok) << "v=" << v;
				EXPECT_EQ(q, expected) << "v=" << v;
			}
		}
	}
}

struct KV {
	int key;
	int value;
};

int compareKV(const KV& a, const KV& b) {
	return a.key - b.key;
}

}


TEST(FrozenBTreeTest, EmptyTreeFindReturnsFalse) {
	ArrayList<int> items;
	ArrayList<char> buffer;
	FrozenBTree<int, 3>::build(buffer, items);

	// An empty tree should still emit a single empty root node.
	ASSERT_EQ(buffer.size(), (int)sizeof(btree_node_t<int, 3>));

	FrozenBTree<int, 3> tree(buffer.getMemory(), compareInts);

	int q = 42;
	EXPECT_FALSE(tree.find(q));
	int q2 = 42;
	EXPECT_FALSE(tree.findNext(q2));
}

TEST(FrozenBTreeTest, SingleItem) {
	ArrayList<int> items;
	items.add(7);
	ArrayList<char> buffer;
	FrozenBTree<int, 3>::build(buffer, items);
	FrozenBTree<int, 3> tree(buffer.getMemory(), compareInts);

	int q = 7;
	EXPECT_TRUE(tree.find(q));
	EXPECT_EQ(q, 7);

	int miss = 8;
	EXPECT_FALSE(tree.find(miss));

	int probe = 6;
	EXPECT_TRUE(tree.findNext(probe));
	EXPECT_EQ(probe, 7);

	int over = 8;
	EXPECT_FALSE(tree.findNext(over));
}

TEST(FrozenBTreeTest, FullSingleLeaf) {
	// Exactly N elements: must fit in the root node, no children.
	runIntRangeChecks<3>(0, 3, 1);
}

TEST(FrozenBTreeTest, JustOverOneLeaf) {
	// N + 1 elements: forces one separator promotion and creates two leaves.
	runIntRangeChecks<3>(0, 4, 1);
}

TEST(FrozenBTreeTest, SmallNMultiLevel) {
	// With N=3 each internal node has up to 4 children; 50 items easily
	// forces multiple internal levels.
	runIntRangeChecks<3>(0, 50, 1);
}

TEST(FrozenBTreeTest, DefaultNLargeDataset) {
	// Default node size with many items; checks lookup over a deep tree.
	runIntRangeChecks<63>(0, 5000, 1);
}

TEST(FrozenBTreeTest, FindNextBetweenItems) {
	// Sparse keys force findNext to actually traverse.
	runIntRangeChecks<3>(0, 200, 5);
}

TEST(FrozenBTreeTest, StructPayload) {
	ArrayList<KV> items;
	for (int i = 0; i < 200; ++i)
		items.add({i * 3, i * 3 + 100});

	ArrayList<char> buffer;
	FrozenBTree<KV, 3>::build(buffer, items);
	FrozenBTree<KV, 3> tree(buffer.getMemory(), compareKV);

	for (int i = 0; i < 200; ++i) {
		KV q{i * 3, 0};
		ASSERT_TRUE(tree.find(q)) << "i=" << i;
		EXPECT_EQ(q.value, i * 3 + 100);
	}

	// Probe between keys: findNext must return the next stored key/value pair.
	for (int i = 0; i < 199; ++i) {
		KV probe{i * 3 + 1, 0};
		ASSERT_TRUE(tree.findNext(probe)) << "i=" << i;
		EXPECT_EQ(probe.key, (i + 1) * 3);
		EXPECT_EQ(probe.value, (i + 1) * 3 + 100);
	}
}

TEST(FrozenBTreeTest, BuildIsMinimalForDenseInput) {
	// With N=3 a perfectly dense tree of 4 elements fits in two nodes
	// (root with 1 separator + 1 leaf child... actually root + 2 leaves of
	// size 2/1 split).  Verify the buffer is reasonably small: at most
	// ceil((M + N) / N) nodes for an M-item input.
	const int M = 100;
	const int N = 3;
	ArrayList<int> items;
	for (int i = 0; i < M; ++i) items.add(i);

	ArrayList<char> buffer;
	FrozenBTree<int, N>::build(buffer, items);

	int nodeCount = buffer.size() / (int)sizeof(btree_node_t<int, N>);
	EXPECT_LE(nodeCount, (M + N) / N + 4) << "tree is not dense enough";

	// Sanity: the resulting tree is still searchable end-to-end.
	FrozenBTree<int, N> tree(buffer.getMemory(), compareInts);
	for (int i = 0; i < M; ++i) {
		int q = i;
		ASSERT_TRUE(tree.find(q));
	}
}

TEST(FrozenBTreeTest, BuildRandomizedThenSorted) {
	// Generate random unique ints, sort them, build, and exhaustively probe.
	std::mt19937 rng(0xC0FFEE);
	std::uniform_int_distribution<int> dist(-100000, 100000);

	ArrayList<int> items;
	std::vector<int> seen;
	while (items.size() < 1500) {
		int v = dist(rng);
		if (std::find(seen.begin(), seen.end(), v) == seen.end()) {
			seen.push_back(v);
			items.add(v);
		}
	}
	std::sort(items.begin(), items.end());

	ArrayList<char> buffer;
	FrozenBTree<int, 7>::build(buffer, items);
	FrozenBTree<int, 7> tree(buffer.getMemory(), compareInts);

	for (int i = 0; i < items.size(); ++i) {
		int q = items.get(i);
		ASSERT_TRUE(tree.find(q)) << "i=" << i << " q=" << q;
	}

	// findNext probes between/around stored values.
	for (int t = 0; t < 1000; ++t) {
		int v = dist(rng);
		int q = v;
		bool ok = tree.findNext(q);

		int expected = INT32_MIN;
		bool any = false;
		for (int i = 0; i < items.size(); ++i) {
			if (items.get(i) >= v) {
				expected = items.get(i);
				any = true;
				break;
			}
		}
		if (!any) {
			EXPECT_FALSE(ok) << "v=" << v;
		} else {
			ASSERT_TRUE(ok) << "v=" << v;
			EXPECT_EQ(q, expected) << "v=" << v;
		}
	}
}

TEST(FrozenBTreeTest, WriteOpsThrow) {
	ArrayList<int> items;
	items.add(1);
	ArrayList<char> buffer;
	FrozenBTree<int, 3>::build(buffer, items);
	FrozenBTree<int, 3> tree(buffer.getMemory(), compareInts);

	// Mutating operations must not be allowed on a frozen tree.
	EXPECT_THROW(tree.insert(2), std::runtime_error);
	EXPECT_THROW(tree.overwrite(1), std::runtime_error);
}
