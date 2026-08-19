/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-02-22
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


#ifndef EXCESSIVE_BTREE_H
#define EXCESSIVE_BTREE_H

#include "fs/FdHandle.h"

#include "stdint.h"
#include "string.h"
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>


/** On-disk / serialized node magic: 'B' | ('T' << 8). */
constexpr uint16_t BTREE_NODE_MAGIC = 0x5442;
/** Current on-disk node format version. */
constexpr uint16_t BTREE_NODE_VERSION = 1;


/**
 * Thrown when a B-Tree node or offset read from untrusted storage is invalid.
 * Callers must treat this as a corrupt (or malicious) tree image.
 */
class BTreeCorruptError : public std::runtime_error {
public:
	explicit BTreeCorruptError(const char* what) : std::runtime_error(what) {}
	explicit BTreeCorruptError(const std::string& what) : std::runtime_error(what) {}
};


/**
 * @brief Header for a B-Tree node.
 * 
 * @tparam N The maximum number of elements in a node.
 */
template <int N = 63>
struct btree_node_header_t {
	uint64_t parent = 0;        /**< Offset of the parent node. */
	int nChildren = 0;          /**< Number of children of this node. */
	int nElements = 0;          /**< Number of elements currently in this node. */
	int indexInParent = -1;     /**< Index of this node in the parent's childOffsets. */

	uint16_t magic = BTREE_NODE_MAGIC;     /**< BTREE_NODE_MAGIC, or 0 for the pre-magic format. */
	uint16_t version = BTREE_NODE_VERSION; /**< BTREE_NODE_VERSION, or 0 for the pre-magic format. */

	uint64_t childOffsets[N + 1] = {}; /**< Offsets of children nodes. */
};


/**
 * @brief A B-Tree node structure.
 * 
 * If sizeof(T) == 24, then sizeof(btree_node_t<T, 63>) == 2048.
 * 
 * @tparam T The type of elements stored in the node.  For building an on-disk
 *           tree map, this must contain both the key and value.  This value
 *           will be used for indexing according to whatever comparator is
 *           selected.
 * @tparam N The maximum number of elements in a node.
 */
template <class T, int N = 63>
struct btree_node_t {
	btree_node_header_t<N> header; /**< Node header containing metadata and child offsets. */
	T elements[N];                 /**< Array of elements in the node. */
};


/**
 * @brief Construct an empty node with valid magic/version.
 */
template <class T, int N = 63>
inline btree_node_t<T, N> makeBTreeNode(uint64_t parent = 0, int nChildren = 0, int nElements = 0, int indexInParent = -1) {
	btree_node_t<T, N> node{};
	node.header.parent = parent;
	node.header.nChildren = nChildren;
	node.header.nElements = nElements;
	node.header.indexInParent = indexInParent;
	node.header.magic = BTREE_NODE_MAGIC;
	node.header.version = BTREE_NODE_VERSION;
	return node;
}


/**
 * @brief Base class for B-Tree implementations.
 * 
 * Note that N is the number of elements, not the number of children per node, which is (N + 1).
 * 
* @tparam T The type of elements stored in the B-Tree.  For building an on-disk
 *          tree map, this must contain both the key and value.  This value
 *          will be used for indexing according to whatever comparator is
 *          selected.
 * @tparam N The maximum number of elements per node.
 */
template <class T, int N = 63>
class BTreeBase {
public:

	/**
	 * Insert a new value to the BTree, if no matching key is found.
	 * If a matching key is found, then no change is made.
	 * @param val value to insert
	 * @return true if the key is already present, false if the key was not previously present and has been inserted
	 */
	bool insert(const T& val) {
		std::unique_lock<std::shared_mutex> lock(mutex);
		btree_node_t<T, N> target;
		btree_node_header_t<N>& header = target.header;
		uint64_t offset;

		int i = findNearest(val, target, offset);
		if (header.nElements > i && compare(target.elements[i], val) == 0)
			return true;

		if (header.nElements == N) {
			// It's full, we need to split it
			btree_node_t<T, N> rhs;
			uint64_t rhsOffset;
			if (offset == getRootOffset())
				rhs = splitRoot(target, rhs, offset, rhsOffset);
			else {
				btree_node_t<T, N> parent = getNode(header.parent);
				rhs = splitChild(parent, target, offset, rhsOffset);
			}

			// > is slightly faster than >=, let's see if you can figure out why :)
			if (i > header.nElements) {
				addToNodeUnchecked(rhs, val, i - header.nElements - 1);
				overwriteNode(rhsOffset, rhs);
			} else {
				addToNodeUnchecked(target, val, i);
				overwriteNode(offset, target);
			}
		} else {
			addToNodeUnchecked(target, val, i);
			overwriteNode(offset, target);
		}

		return false;
	}

	/**
	 * Insert a new value to the BTree, if no matching key is found, and update the existing entry otherwise.
	 * @param val value to insert
	 * @return true if the key was already present and overwritten, false if the key was not previously present and has been newly inserted
	 */
	bool overwrite(const T& val) {
		std::unique_lock<std::shared_mutex> lock(mutex);
		btree_node_t<T, N> target;
		uint64_t offset;

		int i = findNearest(val, target, offset);
		if (target.header.nElements > i && compare(target.elements[i], val) == 0) {
			target.elements[i] = val;
			overwriteNode(offset, target);
			return true;
		}

		if (target.header.nElements == N) {
			// It's full, we need to split it
			btree_node_t<T, N> rhs;
			uint64_t rhsOffset;
			if (offset == getRootOffset())
				rhs = splitRoot(target, rhs, offset, rhsOffset);
			else {
				btree_node_t<T, N> parent = getNode(target.header.parent);
				rhs = splitChild(parent, target, offset, rhsOffset);
			}

			// > is slightly faster than >=, let's see if you can figure out why :)
			if (i > target.header.nElements) {
				addToNodeUnchecked(rhs, val, i - target.header.nElements - 1);
				overwriteNode(rhsOffset, rhs);
			} else {
				addToNodeUnchecked(target, val, i);
				overwriteNode(offset, target);
			}
		} else {
			addToNodeUnchecked(target, val, i);
			overwriteNode(offset, target);
		}

		return false;
	}

	/**
	 * Finds the given value, if present.  Updates val with the value found in the tree, if it is found.
	 * This is useful for constructing key-value stores, where only part of the value is used as the key and is
	 * checked for equivalency, with the other part used as an associated value for that key.
	 * @param val value to search for, updated from the tree if it is present
	 * @return true if the value was found, false if not
	 */
	bool find(T& val) {
		std::shared_lock<std::shared_mutex> lock(mutex);
		btree_node_t<T, N> target;
		uint64_t offset;

		int i = findNearest(val, target, offset);
		if (i < target.header.nElements && compare(target.elements[i], val) == 0) {
			val = target.elements[i];
			return true;
		}

		return false;
	}

	/**
	 * Finds the given value, if present, otherwise the next highest value.  Updates val with the value found in the
	 * tree, if it is found.
	 * @param val value to search for, updated from the tree if it is present
	 * @return true if a match was found, false if not
	 */
	bool findNext(T& val) {
		std::shared_lock<std::shared_mutex> lock(mutex);
		btree_node_t<T, N> node = getRootNode();
		uint64_t offset = getRootOffset();
		bool found = false;
		T best;

		while (true) {
			validateNodeCounts(node.header);

			int low = 0;
			int high = node.header.nElements - 1;
			int midIdx = -1;

			while (low <= high) {
				int mid = (low + high) / 2;
				int c = compare(node.elements[mid], val);
				if (c < 0) {
					low = mid + 1;
				} else if (c == 0) {
					val = node.elements[mid];
					return true;
				} else {
					midIdx = mid;
					high = mid - 1;
				}
			}

			if (midIdx != -1) {
				if (!found || compare(node.elements[midIdx], best) < 0) {
					best = node.elements[midIdx];
					found = true;
				}
			}

			if (node.header.nChildren == 0)
				break;

			if (low < 0 || low >= node.header.nChildren)
				throw BTreeCorruptError("BTree: child index out of range");

			offset = node.header.childOffsets[low];
			node = getNode(offset);
		}

		if (found) {
			val = best;
			return true;
		}
		return false;
	}

	/**
	 * Removes the value from the tree.
	 *
	 * Before descending, a child at `N / 2` keys borrows or merges so a
	 * leaf delete cannot underflow. An internal key is replaced by its
	 * successor. If the root is left with no keys and one child, that
	 * child is copied onto the root offset.
	 *
	 * @param val value to remove
	 * @return true if the value was found and removed, false if not
	 */
	bool remove(T& val) {
		std::unique_lock<std::shared_mutex> lock(mutex);
		return removeFrom(getRootNode(), getRootOffset(), val);
	}

	/**
	 * @brief Destructor for BTreeBase.
	 */
	virtual ~BTreeBase() = default;

protected:
	/**
	 * @brief Constructs a BTreeBase with a custom comparison function.
	 * @param compare Pointer to a function that compares two elements of type T.
	 */
	explicit BTreeBase(int (*compare) (const T&, const T&)) : compare(compare) {}

	/**
	 * Reject nElements / nChildren / indexInParent that cannot be used as
	 * indexes into elements[] or childOffsets[].
	 */
	static void validateNodeCounts(const btree_node_header_t<N>& header) {
		if (header.nElements < 0 || header.nElements > N)
			throw BTreeCorruptError("BTree: nElements out of range");
		if (header.nChildren < 0 || header.nChildren > N + 1)
			throw BTreeCorruptError("BTree: nChildren out of range");
		if (header.nChildren != 0 && header.nChildren != header.nElements + 1)
			throw BTreeCorruptError("BTree: nChildren does not match nElements");
		if (header.indexInParent < -1 || header.indexInParent > N)
			throw BTreeCorruptError("BTree: indexInParent out of range");
	}

	/**
	 * True if this header is a format we can interpret.
	 *
	 * Pre-magic files stored a zero `reserved` word in this slot, so
	 * magic==0 && version==0 is the original on-disk format and must still
	 * be accepted.  New nodes write BTREE_NODE_MAGIC / BTREE_NODE_VERSION.
	 * Any other combination is unknown or corrupt.
	 */
	static bool isRecognizedOnDiskFormat(const btree_node_header_t<N>& header) {
		if (header.magic == 0 && header.version == 0)
			return true;
		return header.magic == BTREE_NODE_MAGIC && header.version == BTREE_NODE_VERSION;
	}

	/**
	 * Full check for a node loaded from untrusted storage (disk or mmap).
	 * Count/index checks run for every recognized format, including legacy.
	 */
	static void validateOnDiskNode(const btree_node_header_t<N>& header) {
		if (!isRecognizedOnDiskFormat(header))
			throw BTreeCorruptError("BTree: invalid magic or version");
		validateNodeCounts(header);
	}

	// The mutex is non-copyable/non-movable but instances of BTreeBase are
	// embedded in other containers (e.g. TreeMap) that rely on copy/move.
	// Each instance gets its own fresh mutex.
	BTreeBase(const BTreeBase& other) : compare(other.compare) {}
	BTreeBase(BTreeBase&& other) noexcept : compare(other.compare) {}
	BTreeBase& operator=(const BTreeBase& other) { compare = other.compare; return *this; }
	BTreeBase& operator=(BTreeBase&& other) noexcept { compare = other.compare; return *this; }

	/**
	 * Finds which node is a good candidate for adding the given value.  The outputted node will always be at the
	 * bottom of the tree, unless the value is already present.  If the value is present in the tree, then this will
	 * find it.
	 * @param val value to look for
	 * @param node output node
	 * @param offset offset of the output node
	 * @return index the value should be inserted at, or the index the value was found at if already present
	 */
	int findNearest(const T& val, btree_node_t<T, N>& node, uint64_t& offset) {
		node = getRootNode();
		offset = getRootOffset();

		while (true) {
			validateNodeCounts(node.header);

			int low = 0;
			int high = node.header.nElements - 1;

			while (low <= high) {
				int mid = (low + high) / 2;
				int c = compare(node.elements[mid], val);

				if (c < 0)
					low = mid + 1;
				else if (c == 0) {
					return mid;
				} else
					high = mid - 1;
			}

			if (node.header.nChildren == 0)
				return low;

			if (low < 0 || low >= node.header.nChildren)
				throw BTreeCorruptError("BTree: child index out of range");

			offset = node.header.childOffsets[low];
			node = getNode(offset);
		}
	}

	static constexpr int minElements() {
		return N / 2;
	}

	int locateKey(const btree_node_t<T, N>& node, const T& val, int& found) const {
		int low = 0;
		int high = node.header.nElements - 1;
		found = -1;
		while (low <= high) {
			int mid = (low + high) / 2;
			int c = compare(node.elements[mid], val);
			if (c < 0)
				low = mid + 1;
			else if (c > 0)
				high = mid - 1;
			else {
				found = mid;
				return mid;
			}
		}
		return low;
	}

	T minInSubtree(uint64_t offset) {
		btree_node_t<T, N> node = getNode(offset);
		while (node.header.nChildren != 0) {
			validateNodeCounts(node.header);
			offset = node.header.childOffsets[0];
			node = getNode(offset);
		}
		return node.elements[0];
	}

	void writeChildHeader(uint64_t childOff, uint64_t parentOff, int indexInParent) {
		btree_node_t<T, N> child = getNode(childOff);
		child.header.parent = parentOff;
		child.header.indexInParent = indexInParent;
		overwriteNodeHeader(childOff, child);
	}

	void reindexChildren(const btree_node_t<T, N>& node, uint64_t nodeOff, int from) {
		for (int j = from; j < node.header.nChildren; j++)
			writeChildHeader(node.header.childOffsets[j], nodeOff, j);
	}

	void absorbOnlyChild(btree_node_t<T, N>& node, uint64_t offset) {
		uint64_t parentOff = node.header.parent;
		int indexInParent = node.header.indexInParent;
		btree_node_t<T, N> child = getNode(node.header.childOffsets[0]);
		child.header.parent = parentOff;
		child.header.indexInParent = indexInParent;
		reindexChildren(child, offset, 0);
		overwriteNode(offset, child);
		node = child;
	}

	void borrowFromSibling(btree_node_t<T, N>& parent, uint64_t parentOff, int i, int sib) {
		uint64_t childOff = parent.header.childOffsets[i];
		uint64_t sibOff = parent.header.childOffsets[sib];
		btree_node_t<T, N> child = getNode(childOff);
		btree_node_t<T, N> sibling = getNode(sibOff);
		int sep = sib < i ? sib : i;

		if (sib < i) {
			for (int j = child.header.nElements - 1; j >= 0; j--)
				child.elements[j + 1] = child.elements[j];
			child.elements[0] = parent.elements[sep];
			child.header.nElements++;
			if (child.header.nChildren > 0) {
				for (int j = child.header.nChildren - 1; j >= 0; j--)
					child.header.childOffsets[j + 1] = child.header.childOffsets[j];
				uint64_t moved = sibling.header.childOffsets[sibling.header.nChildren - 1];
				child.header.childOffsets[0] = moved;
				child.header.nChildren++;
				sibling.header.nChildren--;
				writeChildHeader(moved, childOff, 0);
				reindexChildren(child, childOff, 1);
			}
			parent.elements[sep] = sibling.elements[sibling.header.nElements - 1];
			sibling.header.nElements--;
		} else {
			child.elements[child.header.nElements] = parent.elements[sep];
			child.header.nElements++;
			if (child.header.nChildren > 0) {
				uint64_t moved = sibling.header.childOffsets[0];
				child.header.childOffsets[child.header.nChildren] = moved;
				child.header.nChildren++;
				writeChildHeader(moved, childOff, child.header.nChildren - 1);
				for (int j = 1; j < sibling.header.nChildren; j++)
					sibling.header.childOffsets[j - 1] = sibling.header.childOffsets[j];
				sibling.header.nChildren--;
				reindexChildren(sibling, sibOff, 0);
			}
			parent.elements[sep] = sibling.elements[0];
			for (int j = 1; j < sibling.header.nElements; j++)
				sibling.elements[j - 1] = sibling.elements[j];
			sibling.header.nElements--;
		}

		overwriteNode(sibOff, sibling);
		overwriteNode(childOff, child);
		overwriteNode(parentOff, parent);
	}

	void mergeChildren(btree_node_t<T, N>& parent, uint64_t parentOff, int i) {
		uint64_t leftOff = parent.header.childOffsets[i];
		uint64_t rightOff = parent.header.childOffsets[i + 1];
		btree_node_t<T, N> left = getNode(leftOff);
		btree_node_t<T, N> right = getNode(rightOff);

		left.elements[left.header.nElements] = parent.elements[i];
		for (int j = 0; j < right.header.nElements; j++)
			left.elements[left.header.nElements + 1 + j] = right.elements[j];
		int oldLeftChildren = left.header.nChildren;
		if (right.header.nChildren > 0) {
			for (int j = 0; j < right.header.nChildren; j++) {
				left.header.childOffsets[oldLeftChildren + j] = right.header.childOffsets[j];
				writeChildHeader(right.header.childOffsets[j], leftOff, oldLeftChildren + j);
			}
			left.header.nChildren = oldLeftChildren + right.header.nChildren;
		}
		left.header.nElements += 1 + right.header.nElements;
		overwriteNode(leftOff, left);

		for (int j = i + 1; j < parent.header.nElements; j++)
			parent.elements[j - 1] = parent.elements[j];
		parent.header.nElements--;
		for (int j = i + 2; j < parent.header.nChildren; j++)
			parent.header.childOffsets[j - 1] = parent.header.childOffsets[j];
		parent.header.nChildren--;
		reindexChildren(parent, parentOff, i + 1);

		if (parentOff == getRootOffset() && parent.header.nElements == 0 && parent.header.nChildren == 1)
			absorbOnlyChild(parent, parentOff);
		else
			overwriteNode(parentOff, parent);
	}

	void ensureChildCanLoseKey(btree_node_t<T, N>& parent, uint64_t parentOff, int i) {
		if (getNode(parent.header.childOffsets[i]).header.nElements > minElements())
			return;
		if (i > 0 && getNode(parent.header.childOffsets[i - 1]).header.nElements > minElements())
			borrowFromSibling(parent, parentOff, i, i - 1);
		else if (i + 1 < parent.header.nChildren &&
		         getNode(parent.header.childOffsets[i + 1]).header.nElements > minElements())
			borrowFromSibling(parent, parentOff, i, i + 1);
		else
			mergeChildren(parent, parentOff, i > 0 ? i - 1 : i);
	}

	bool removeFrom(btree_node_t<T, N> node, uint64_t offset, const T& val) {
		validateNodeCounts(node.header);

		int found = -1;
		int idx = locateKey(node, val, found);

		if (found >= 0 && node.header.nChildren == 0) {
			for (int j = found + 1; j < node.header.nElements; j++)
				node.elements[j - 1] = node.elements[j];
			node.header.nElements--;
			overwriteNode(offset, node);
			return true;
		}
		if (node.header.nChildren == 0)
			return false;

		if (found >= 0) {
			uint64_t rightOff = node.header.childOffsets[found + 1];
			if (getNode(rightOff).header.nElements <= minElements()) {
				ensureChildCanLoseKey(node, offset, found + 1);
				return removeFrom(getNode(offset), offset, val);
			}
			T succ = minInSubtree(rightOff);
			node.elements[found] = succ;
			overwriteNode(offset, node);
			return removeFrom(getNode(rightOff), rightOff, succ);
		}

		int oldElements = node.header.nElements;
		int oldChildren = node.header.nChildren;
		ensureChildCanLoseKey(node, offset, idx);
		node = getNode(offset);
		if (node.header.nElements != oldElements || node.header.nChildren != oldChildren)
			return removeFrom(node, offset, val);
		uint64_t childOff = node.header.childOffsets[idx];
		return removeFrom(getNode(childOff), childOff, val);
	}

	/**
	 * @brief Shifts elements and inserts a new element into a node without checking for capacity.
	 * @param node The node to modify.
	 * @param element The element to insert.
	 * @param idx The index at which to insert the element.
	 */
	void addToNodeUnchecked(btree_node_t<T, N>& node, const T& element, int idx) {
		if (idx < 0 || idx > node.header.nElements || node.header.nElements < 0 || node.header.nElements >= N)
			throw BTreeCorruptError("BTree: insert index out of range");
		for (int j = node.header.nElements - 1; j >= idx; j--)
			node.elements[j + 1] = node.elements[j];
		new (&node.elements[idx]) T(element);
		node.header.nElements++;
	}

	/**
	 * @brief Recursively inserts an element into a node that is not full.
	 * @param node The node to insert into.
	 * @param offset The disk offset of the node.
	 * @param element The element to insert.
	 */
	void insertNonFull(btree_node_t<T, N>& node, uint64_t offset, const T& element) {
		validateNodeCounts(node.header);

		int i = scanNode(node, element);

		if (node.header.nChildren == 0) {
			if (i < 0 || i > node.header.nElements || node.header.nElements >= N)
				throw BTreeCorruptError("BTree: insert index out of range");
			memmove(&node.elements[i + 1], &node.elements[i], (node.header.nElements - i) * sizeof(T));
			node.elements[i] = element;
			node.header.nElements++;
			overwriteNode(offset, node);

		} else {
			if (i < 0 || i >= node.header.nChildren)
				throw BTreeCorruptError("BTree: child index out of range");
			btree_node_t<T, N> child = getNode(node.header.childOffsets[i]);

			if (child.header.nElements == N) {
				uint64_t newChildOffset;
				btree_node_t<T, N> newChild = splitChild(node, child, node.header.childOffsets[i], newChildOffset);
				if (compare(node.elements[i], element) <= 0) {
					child = newChild;
					i++;
				}
			}

			if (i < 0 || i >= node.header.nChildren)
				throw BTreeCorruptError("BTree: child index out of range");
			insertNonFull(child, node.header.childOffsets[i], element);
		}

	}

	/**
	 * @brief Splits the root node of the B-Tree.
	 * @param oldRoot The current root node.
	 * @param newRoot The new root node being created.
	 * @param oldRootOffset Output parameter for the new offset of the old root.
	 * @param newChildOffset Output parameter for the offset of the new child.
	 * @return The new child node created by splitting.
	 */
	virtual btree_node_t<T, N> splitRoot(btree_node_t<T, N>& oldRoot, btree_node_t<T, N>& newRoot, uint64_t& oldRootOffset, uint64_t& newChildOffset) {
		oldRoot.header.parent = getRootOffset();
		oldRoot.header.indexInParent = 0;
		oldRootOffset = addNode(oldRoot);

		// Update child references to the root
		for (int j = 0; j < oldRoot.header.nChildren; j++) {
			btree_node_t<T, N> subChild = getNode(oldRoot.header.childOffsets[j]);
			subChild.header.parent = oldRootOffset;
			overwriteNodeHeader(oldRoot.header.childOffsets[j], subChild);
		}

		newRoot = makeBTreeNode<T, N>(0, 1, 0, -1);
		newRoot.header.childOffsets[0] = oldRootOffset;

		return splitChild(newRoot, oldRoot, oldRootOffset, newChildOffset);
	}

	/**
	 * @brief Splits a child node when it becomes full.
	 * @param parent The parent node.
	 * @param originalChild The full child node to split.
	 * @param childOffset The offset of the child node.
	 * @param newChildOffset Output parameter for the offset of the newly created node.
	 * @return The newly created node.
	 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winfinite-recursion"
#endif
	btree_node_t<T, N> splitChild(btree_node_t<T, N>& parent, btree_node_t<T, N>& originalChild, uint64_t childOffset, uint64_t& newChildOffset) {
		if (parent.header.nElements == N) {
			btree_node_t<T, N> rhs;
			uint64_t rhsOffset;
			if (originalChild.header.parent == getRootOffset())
				rhs = splitRoot(parent, rhs, originalChild.header.parent, rhsOffset);
			else {
				btree_node_t<T, N> parent2 = getNode(parent.header.parent);
				rhs = splitChild(parent2, parent, originalChild.header.parent, rhsOffset);
			}

			if (originalChild.header.indexInParent > parent.header.nElements) {
				originalChild.header.indexInParent -= parent.header.nElements + 1;
				originalChild.header.parent = rhsOffset;
				parent = rhs;
			}
		}

		validateNodeCounts(parent.header);
		validateNodeCounts(originalChild.header);
		if (originalChild.header.nElements != N)
			throw BTreeCorruptError("BTree: split of non-full node");

		int i = originalChild.header.indexInParent;
		if (i < 0 || i > parent.header.nElements)
			throw BTreeCorruptError("BTree: indexInParent out of range");

		btree_node_t<T, N> newChild = {{originalChild.header.parent, 0, (originalChild.header.nElements + 1) / 2 - 1, i + 1, 0, {}}, {}};

		// Update the size of the original child, which is now much smaller
		originalChild.header.nElements = originalChild.header.nElements - newChild.header.nElements - 1;

		// Copy over the elements
		for (int j = 0; j < newChild.header.nElements; j++)
			newChild.elements[j] = originalChild.elements[originalChild.header.nElements + j + 1];

		if (originalChild.header.nChildren > 0) {
			// Copy over the children
			newChild.header.nChildren = newChild.header.nElements + 1;
			originalChild.header.nChildren -= newChild.header.nChildren;
			memcpy(newChild.header.childOffsets, &originalChild.header.childOffsets[originalChild.header.nChildren], sizeof(uint64_t) * newChild.header.nChildren);
		}

		// We need to add the new child so we can get the offset of it
		newChildOffset = addNode(newChild);

		for (int j = parent.header.nElements - i - 1; j >= 0; j--)
			parent.header.childOffsets[i + j + 2] = parent.header.childOffsets[i + j + 1];
		parent.header.childOffsets[i + 1] = newChildOffset;

		for (int j = parent.header.nElements - i - 1; j >= 0; j--)
			parent.elements[i + j + 1] = parent.elements[i + j];
		parent.elements[i] = originalChild.elements[originalChild.header.nElements];

		parent.header.nElements++;
		parent.header.nChildren++;

		// Update any children that got moved in the parent
		for (int j = i + 2; j < parent.header.nChildren; j++) {
			btree_node_t<T, N> subChild = getNode(parent.header.childOffsets[j]);
			subChild.header.indexInParent = j;
			overwriteNodeHeader(parent.header.childOffsets[j], subChild);
		}

		// Update any children that got moved to newChild
		for (int j = 0; j < newChild.header.nChildren; j++) {
			btree_node_t<T, N> subChild = getNode(newChild.header.childOffsets[j]);
			subChild.header.indexInParent = j;
			subChild.header.parent = newChildOffset;
			overwriteNodeHeader(newChild.header.childOffsets[j], subChild);
		}

		overwriteNodeHeader(childOffset, originalChild);
		overwriteNode(originalChild.header.parent, parent);

		return newChild;
	}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

	/**
	 * @brief Scans a node to find the position for an element.
	 * @param node The node to scan.
	 * @param element The element to look for.
	 * @return The index where the element is located or should be inserted.
	 */
	int scanNode(const btree_node_t<T, N>& node, const T& element) {
		int i = 0;
		for (; i < node.header.nElements; i++) {
			if (compare(node.elements[i], element) >= 0)
				break;
		}

		return i;
	}

	/**
	 * @brief Retrieves a node from storage at the specified offset.
	 * @param offset The disk offset of the node.
	 * @return The node read from storage.
	 */
	virtual btree_node_t<T, N> getNode(uint64_t offset) const = 0;

	/**
	 * @brief Retrieves the root node of the B-Tree.
	 * @return The root node.
	 */
	virtual btree_node_t<T, N> getRootNode() const = 0;

	/**
	 * @brief Retrieves the disk offset of the root node.
	 * @return The offset of the root node.
	 */
	virtual uint64_t getRootOffset() const = 0;

	/**
	 * @brief Adds a new node to storage.
	 * @param node The node to add.
	 * @return The disk offset where the node was added.
	 */
	virtual uint64_t addNode(const btree_node_t<T, N>& node) = 0;

	/**
	 * @brief Overwrites an existing node in storage.
	 * @param offset The disk offset of the node.
	 * @param node The node data to write.
	 */
	virtual void overwriteNode(uint64_t offset, const btree_node_t<T, N>& node) = 0;

	/**
	 * @brief Overwrites only the header of an existing node in storage.
	 * 
	 * This just overwrites the initial data, currently the first three longs.
	 * Should save time if the children and elements have not been modified.
	 * 
	 * @param offset The disk offset of the node.
	 * @param node The node containing the header to write.
	 */
	virtual void overwriteNodeHeader(uint64_t offset, const btree_node_t<T, N>& node) = 0;

	int (*compare) (const T&, const T&); /**< Comparison function for elements of type T. */

	mutable std::shared_mutex mutex;
};


/**
 * This is a lightweight and threadsafe BTree implementation.
 * Note that it is designed for memory efficiency over speed.
 * @tparam T
 * @tparam N
 */
template <class T, int N = 63>
class BTree: public BTreeBase<T, N> {
public:
	/**
	 * @brief Constructs a BTree with a file handle and root offset.
	 * @param file The file handle to use for storage.
	 * @param rootOffset The disk offset of the root node.
	 * @param compare Comparison function for elements.
	 */
	BTree(FdHandle&& file, off_t rootOffset, int (*compare) (const T&, const T&))
		: BTreeBase<T, N>(compare), rootOffset(rootOffset), file(file) {

		if (rootOffset < 0)
			throw BTreeCorruptError("BTree: negative root offset");
		if (file.isNew())
			initialize();
		else
			(void)getNode((uint64_t)rootOffset);
	}

	/**
	 * @brief Constructs a BTree with a file handle and root offset (copy version).
	 * @param file The file handle to use for storage.
	 * @param rootOffset The disk offset of the root node.
	 * @param compare Comparison function for elements.
	 */
	BTree(const FdHandle& file, off_t rootOffset, int (*compare) (const T&, const T&))
			: BTreeBase<T, N>(compare), rootOffset(rootOffset), file(file) {

		if (rootOffset < 0)
			throw BTreeCorruptError("BTree: negative root offset");
		if (file.isNew())
			initialize();
		else
			(void)getNode((uint64_t)rootOffset);
	}

	/**
	 * @brief Gets the offset immediately following the root node.
	 * @return The offset after the root node.
	 */
	off_t getHeaderEndOffset() const {
		return rootOffset + sizeof(btree_node_t<T, N>);
	}

	/**
	 * @brief Initializes a new B-Tree by writing an empty root node.
	 */
	void initialize() {
		btree_node_t<T, N> root = makeBTreeNode<T, N>();
		file.pwrite(root, rootOffset);
	}

protected:

	/**
	 * @brief Retrieves a node from the file.
	 * @param offset Disk offset.
	 * @return The node.
	 */
	btree_node_t<T, N> getNode(uint64_t offset) const override {
		checkNodeOffset(offset);
		btree_node_t<T, N> resultat{};
		ssize_t n = file.pread(resultat, (off_t)offset);
		if (n != (ssize_t)sizeof(resultat))
			throw BTreeCorruptError("BTree: short read");
		this->validateOnDiskNode(resultat.header);
		return resultat;
	}

	/**
	 * @brief Retrieves the root node from the file.
	 * @return The root node.
	 */
	btree_node_t<T, N> getRootNode() const override {
		return getNode(rootOffset);
	}

	/**
	 * @brief Returns the root offset.
	 * @return The root offset.
	 */
	uint64_t getRootOffset() const override {
		return rootOffset;
	}

	/**
	 * @brief Appends a new node to the file.
	 * @param node The node to add.
	 * @return The offset where it was added.
	 */
	uint64_t addNode(const btree_node_t<T, N>& node) override {
		uint64_t offset = file.seekToEndWithPadding(8);
		file.pwrite(node, offset);
		return offset;
	}

	/**
	 * @brief Overwrites a node in the file.
	 * @param offset Disk offset.
	 * @param node Node data.
	 */
	void overwriteNode(uint64_t offset, const btree_node_t<T, N>& node) override {
		checkNodeOffset(offset);
		file.pwrite(node, (off_t)offset);
	}

	/**
	 * @brief Overwrites only the header of a node in the file.
	 * @param offset Disk offset.
	 * @param node Node data containing the header.
	 */
	void overwriteNodeHeader(uint64_t offset, const btree_node_t<T, N>& node) override {
		checkNodeOffset(offset);
		file.pwrite(&node, sizeof(node.header), (off_t)offset);
	}

	uint64_t currentFileSize() const {
		struct stat st;
		if (fstat(file.getFd(), &st) != 0)
			throw BTreeCorruptError("BTree: fstat failed");
		if (st.st_size < 0)
			throw BTreeCorruptError("BTree: invalid file size");
		return (uint64_t)st.st_size;
	}

	void checkNodeOffset(uint64_t offset) const {
		const uint64_t nodeSize = sizeof(btree_node_t<T, N>);
		uint64_t size = currentFileSize();
		if (offset > size || nodeSize > size - offset)
			throw BTreeCorruptError("BTree: node offset out of range");
	}

	off_t rootOffset; /**< Offset of the root node in the file. */
	FdHandle file;    /**< File handle for storage. */
};

#endif //EXCESSIVE_BTREE_H
