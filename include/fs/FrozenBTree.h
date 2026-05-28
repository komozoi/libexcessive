/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-05-27
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


#ifndef EXCESSIVE_FROZEN_BTREE_H
#define EXCESSIVE_FROZEN_BTREE_H

#include <stdexcept>

#include "ds/ArrayList.h"
#include "fs/BTree.h"


/**
 * Provides a BTree from a section of memory.
 * This is compatible with mmap, allowing a section of a file to be used as a BTree for read-only access.
 * This is useful for constructing read-only segments of an index.
 * @tparam T
 * @tparam N
 */
template <class T, int N = 63>
class FrozenBTree: public BTreeBase<T, N> {
public:
	/**
	 * @brief Constructs a BTree with a memory region and root offset.
	 * @param memory Pointer to the memory region to use as the BTree.
	 * @param compare Comparison function for elements.
	 */
	FrozenBTree(const char* memory, int (*compare) (const T&, const T&))
		: BTreeBase<T, N>(compare), memory(memory) {
	}

	/**
	 * Builds an optimized BTree from a container of items.  This prepares it for later searching.
	 *
	 * Items must already be sorted according to the same comparator that will be used at query time.
	 * The serialized nodes are appended to dst in breadth-first order so the root node sits at the
	 * start of dst (offset 0 relative to dst.getMemory()).
	 *
	 * @tparam U Source container type, supporting size() and operator[] / get(i) via begin()/end().
	 * @param dst Destination buffer to write the encoded btree to
	 * @param items Source container of items to put in the BTree (must be pre-sorted)
	 */
	template <class U>
	static void build(ArrayList<char>& dst, const U& items) {
		typedef btree_node_t<T, N> node_t;

		// Copy input into a contiguous array so we can index it.
		ArrayList<T> sortedItems;
		for (const T& it : items)
			sortedItems.add(it);
		int M = sortedItems.size();

		// In-memory representation of the tree before serialization.  Each
		// pending node holds its final node_t payload and the indices (into
		// `pending`) of its children.  Offsets are assigned in a second pass.
		struct PendingNode {
			node_t node;
			ArrayList<int> childIndices;
			uint64_t offset;
		};

		ArrayList<PendingNode> pending;
		ArrayList<int> currentLevel;
		ArrayList<T> currentSeparators;

		// Build the leaf level greedily: up to N items per leaf, with a
		// single item between consecutive leaves promoted as a separator.
		// Special case: when exactly N+1 items remain we cannot take a full
		// leaf, since that would promote the single trailing item with no
		// next leaf to pair it with.  In that case shrink the current leaf
		// so the remainder splits cleanly into (separator + next leaf).
		int i = 0;
		do {
			int remaining = M - i;
			int take;
			if (remaining <= N)
				take = remaining;
			else if (remaining == N + 1)
				take = N - 1;
			else
				take = N;

			PendingNode pn;
			pn.node = node_t{};
			pn.offset = 0;
			pn.node.header.nElements = take;
			pn.node.header.nChildren = 0;
			for (int j = 0; j < take; j++)
				pn.node.elements[j] = sortedItems.get(i + j);
			pending.add(std::move(pn));
			currentLevel.add(pending.size() - 1);
			i += take;
			if (i < M) {
				currentSeparators.add(sortedItems.get(i));
				i++;
			}
		} while (i < M);

		// Build internal levels until a single root remains.  Each internal
		// node groups up to N+1 children with N separators consumed between
		// them; one additional separator is then promoted up between
		// consecutive parents at this level.
		while (currentLevel.size() > 1) {
			ArrayList<int> nextLevel;
			ArrayList<T> nextSeparators;
			int kids = currentLevel.size();
			int idx = 0;
			int sepIdx = 0;
			while (idx < kids) {
				int remaining = kids - idx;
				int childCount = (N + 1) < remaining ? (N + 1) : remaining;

				PendingNode pn;
				pn.node = node_t{};
				pn.offset = 0;
				pn.node.header.nChildren = childCount;
				pn.node.header.nElements = childCount - 1;
				for (int c = 0; c < childCount; c++)
					pn.childIndices.add(currentLevel.get(idx + c));
				for (int s = 0; s < childCount - 1; s++)
					pn.node.elements[s] = currentSeparators.get(sepIdx + s);
				pending.add(std::move(pn));
				nextLevel.add(pending.size() - 1);

				idx += childCount;
				sepIdx += childCount - 1;
				if (idx < kids) {
					nextSeparators.add(currentSeparators.get(sepIdx));
					sepIdx++;
				}
			}
			currentLevel = std::move(nextLevel);
			currentSeparators = std::move(nextSeparators);
		}

		int rootIdx = currentLevel.get(0);

		// Assign final byte offsets in breadth-first order so the root lands
		// at offset 0.  Then patch each node's childOffsets / parent /
		// indexInParent based on the assigned offsets.
		ArrayList<int> bfs;
		bfs.add(rootIdx);
		int head = 0;
		while (head < bfs.size()) {
			int idx = bfs.get(head++);
			for (int c = 0; c < pending.get(idx).childIndices.size(); c++)
				bfs.add(pending.get(idx).childIndices.get(c));
		}
		for (int k = 0; k < bfs.size(); k++)
			pending.get(bfs.get(k)).offset = (uint64_t)k * sizeof(node_t);

		pending.get(rootIdx).node.header.parent = 0;
		pending.get(rootIdx).node.header.indexInParent = -1;
		for (int k = 0; k < bfs.size(); k++) {
			PendingNode& pn = pending.get(bfs.get(k));
			for (int c = 0; c < pn.childIndices.size(); c++) {
				PendingNode& child = pending.get(pn.childIndices.get(c));
				pn.node.header.childOffsets[c] = child.offset;
				child.node.header.parent = pn.offset;
				child.node.header.indexInParent = c;
			}
		}

		// Serialize in BFS order.
		for (int k = 0; k < bfs.size(); k++) {
			const node_t& n = pending.get(bfs.get(k)).node;
			dst.addMany(reinterpret_cast<const char*>(&n), (int)sizeof(node_t));
		}
	}

protected:

	/**
	 * @brief Retrieves a node from the memory region.
	 * @param offset Memory offset.
	 * @return The node.
	 */
	btree_node_t<T, N> getNode(uint64_t offset) const override {
		return *(btree_node_t<T, N>*)&memory[offset];
	}

	/**
	 * @brief Retrieves the root node from the memory region.
	 * @return The root node.
	 */
	btree_node_t<T, N> getRootNode() const override {
		return *(btree_node_t<T, N>*)memory;
	}

	/**
	 * @brief Returns the root offset.
	 * @return The root offset.
	 */
	uint64_t getRootOffset() const override {
		return 0;
	}

	/**
	 * @brief Appends a new node to the memory region.
	 * @param node The node to add.
	 * @return The offset where it was added.
	 */
	uint64_t addNode(const btree_node_t<T, N>& node) override {
		throw std::runtime_error("Cannot add node to frozen BTree");
	}

	/**
	 * @brief Overwrites a node in the memory region.
	 * @param offset Memory offset.
	 * @param node Node data.
	 */
	void overwriteNode(uint64_t offset, const btree_node_t<T, N>& node) override {
		throw std::runtime_error("Cannot overwrite node in frozen BTree");
	}

	/**
	 * @brief Overwrites only the header of a node in the memory region.
	 * @param offset Memory offset.
	 * @param node Node data containing the header.
	 */
	void overwriteNodeHeader(uint64_t offset, const btree_node_t<T, N>& node) override {
		throw std::runtime_error("Cannot overwrite node header in frozen BTree");
	}

	const char* memory;
};

#endif //EXCESSIVE_FROZEN_BTREE_H
