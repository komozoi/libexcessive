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
#include "stddef.h"
#include "string.h"


/**
 * @brief Header for a B-Tree node.
 * 
 * @tparam N The maximum number of elements in a node.
 */
template <int N = 63>
struct btree_node_header_t {
	uint64_t parent;        /**< Offset of the parent node. */
	int nChildren;          /**< Number of children of this node. */
	int nElements;          /**< Number of elements currently in this node. */
	int indexInParent;      /**< Index of this node in the parent's childOffsets. */

	// Unused 4 bytes for 8-byte alignment
	uint32_t reserved;      /**< Reserved for 8-byte alignment. */

	uint64_t childOffsets[N + 1]; /**< Offsets of children nodes. */
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
	/**
	 * @brief Default constructor for btree_node_t.
	 */
	btree_node_t() = default;

	btree_node_header_t<N> header; /**< Node header containing metadata and child offsets. */
	T elements[N];                 /**< Array of elements in the node. */
};


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
		btree_node_t<T, N> node;
		uint64_t offset;

		int i = findNearest(val, node, offset);
		if (i > 0)
			i--;

		while (true) {
			if (i < node.header.nElements && compare(node.elements[i], val) >= 0) {
				// Found a valid next value (either equal or next greater)
				val = node.elements[i];
				return true;
			}

			// If this node has no children, there is no next element
			if (node.header.nChildren == 0)
				return false;

			// Go to the next child
			offset = node.header.childOffsets[i];
			node = getNode(offset);
			i = 0;
		}
	}

	/**
	 * Removes the value from the tree
	 * TODO: Make this function rebalance tree correctly
	 * Currently this function does NOT rebalance the tree.
	 * @param val value to remove
	 * @return true if the value was found and removed, false if not
	 */
	bool remove(T& val) {
		btree_node_t<T, N> node;
		uint64_t offset;

		int i = findNearest(val, node, offset);
		if (i >= node.header.nElements || compare(node.elements[i], val) != 0)
			return false;

		// Only remove from leaf nodes in this basic version
		if (node.header.nChildren != 0) {
			// Full B-tree remove (merging, rotation, etc.) would be required here
			return false;
		}

		// Remove element
		for (int j = i + 1; j < node.header.nElements; j++)
			node.elements[j - 1] = node.elements[j];
		node.header.nElements--;

		overwriteNode(offset, node);
		return true;
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
			int low = 0;
			int high = node.header.nElements - 1;

			while (low <= high) {
				int mid = (low + high) / 2;
				int c = compare(node.elements[mid], val);

				if (c < 0)
					high = mid - 1;
				else if (c == 0) {
					return mid;
				} else
					low = mid + 1;
			}

			if (node.header.nChildren == 0)
				return low;

			offset = node.header.childOffsets[low];
			node = getNode(offset);
		}
	}

	/**
	 * @brief Shifts elements and inserts a new element into a node without checking for capacity.
	 * @param node The node to modify.
	 * @param element The element to insert.
	 * @param idx The index at which to insert the element.
	 */
	void addToNodeUnchecked(btree_node_t<T, N>& node, const T& element, int idx) {
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
		int i = scanNode(node, element);

		if (node.header.nChildren == 0) {
			memmove(&node.elements[i + 1], &node.elements[i], (node.header.nElements - i) * sizeof(T));
			node.elements[i] = element;
			node.header.nElements++;
			overwriteNode(offset, node);

		} else {
			btree_node_t<T, N> child = getNode(node.childOffsets[i]);

			if (child.header.nElements == N) {
				uint64_t newChildOffset;
				btree_node_t<T, N> newChild = splitChild(node, child, node.childOffsets[i], newChildOffset);
				if (compare(node.elements[i], element) <= 0) {
					child = newChild;
					i++;
				}
			}

			insertNonFull(child, node.childOffsets[i], element);
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

		newRoot = {{0, 1, 0, -1, 0, {}}, {}};
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

		int i = originalChild.header.indexInParent;
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

	/**
	 * @brief Scans a node to find the position for an element.
	 * @param node The node to scan.
	 * @param element The element to look for.
	 * @return The index where the element is located or should be inserted.
	 */
	int scanNode(const btree_node_t<T, N>& node, const T& element) {
		int i = 0;
		for (; i < node.header.nElements; i++) {
			if (compare(node.elements[i], element) <= 0)
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

		if (file.isNew())
			initialize();
	}

	/**
	 * @brief Constructs a BTree with a file handle and root offset (copy version).
	 * @param file The file handle to use for storage.
	 * @param rootOffset The disk offset of the root node.
	 * @param compare Comparison function for elements.
	 */
	BTree(const FdHandle& file, off_t rootOffset, int (*compare) (const T&, const T&))
			: BTreeBase<T, N>(compare), rootOffset(rootOffset), file(file) {

		if (file.isNew())
			initialize();
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
		btree_node_t<T, N> root{{0, 0, 0, -1, 0, {}}, {}};

		file.seek(rootOffset);
		file.write(root);
	}

protected:

	/**
	 * @brief Retrieves a node from the file.
	 * @param offset Disk offset.
	 * @return The node.
	 */
	btree_node_t<T, N> getNode(uint64_t offset) const override {
		btree_node_t<T, N> resultat;

		file.seek(offset, SEEK_SET);
		file.read(resultat);
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
		file.write(node);
		return offset;
	}

	/**
	 * @brief Overwrites a node in the file.
	 * @param offset Disk offset.
	 * @param node Node data.
	 */
	void overwriteNode(uint64_t offset, const btree_node_t<T, N>& node) override {
		file.seek(offset);
		file.write(node);
	}

	/**
	 * @brief Overwrites only the header of a node in the file.
	 * @param offset Disk offset.
	 * @param node Node data containing the header.
	 */
	void overwriteNodeHeader(uint64_t offset, const btree_node_t<T, N>& node) override {
		file.seek(offset);
		file.write(&node, sizeof(node.header));
	}

	off_t rootOffset; /**< Offset of the root node in the file. */
	FdHandle file;    /**< File handle for storage. */
};

#endif //EXCESSIVE_BTREE_H
