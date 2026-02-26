//
// Created by nuclaer on 22.2.25.
//

#ifndef EXCESSIVE_BTREE_H
#define EXCESSIVE_BTREE_H

#include "fs/FdHandle.h"

#include "stdint.h"
#include "stddef.h"
#include "string.h"


template <int N = 63>
struct btree_node_header_t {
	uint64_t parent;
	int nChildren;
	int nElements;
	int indexInParent;

	// Unused 4 bytes for 8-byte alignment
	uint32_t reserved;

	uint64_t childOffsets[N + 1];
};


/**
 * If sizeof(T) == 24, then sizeof(btree_node_t<T, 63>) == 2048.
*/
template <class T, int N = 63>
struct btree_node_t {
	btree_node_t() = default;

	btree_node_header_t<N> header;
	T elements[N];
};


// Note that N is the number of elements, not the number of children per node, which is (N + 1).
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

	virtual ~BTreeBase() = default;

protected:
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

	void addToNodeUnchecked(btree_node_t<T, N>& node, const T& element, int idx) {
		for (int j = node.header.nElements - 1; j >= idx; j--)
			node.elements[j + 1] = node.elements[j];
		new (&node.elements[idx]) T(element);
		node.header.nElements++;
	}

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

	int scanNode(const btree_node_t<T, N>& node, const T& element) {
		int i = 0;
		for (; i < node.header.nElements; i++) {
			if (compare(node.elements[i], element) <= 0)
				break;
		}

		return i;
	}

	virtual btree_node_t<T, N> getNode(uint64_t offset) const = 0;
	virtual btree_node_t<T, N> getRootNode() const = 0;
	virtual uint64_t getRootOffset() const = 0;

	virtual uint64_t addNode(const btree_node_t<T, N>& node) = 0;
	virtual void overwriteNode(uint64_t offset, const btree_node_t<T, N>& node) = 0;

	// This just overwrites the initial data, currently the first three longs.  Should save time
	// if the children and elements have not been modified.
	virtual void overwriteNodeHeader(uint64_t offset, const btree_node_t<T, N>& node) = 0;

	int (*compare) (const T&, const T&);
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
	BTree(FdHandle&& file, off_t rootOffset, int (*compare) (const T&, const T&))
		: BTreeBase<T, N>(compare), rootOffset(rootOffset), file(file) {

		if (file.isNew())
			initialize();
	}

	BTree(const FdHandle& file, off_t rootOffset, int (*compare) (const T&, const T&))
			: BTreeBase<T, N>(compare), rootOffset(rootOffset), file(file) {

		if (file.isNew())
			initialize();
	}

	off_t getHeaderEndOffset() const {
		return rootOffset + sizeof(btree_node_t<T, N>);
	}

	void initialize() {
		btree_node_t<T, N> root{{0, 0, 0, -1, 0, {}}, {}};

		file.seek(rootOffset);
		file.write(root);
	}

protected:

	btree_node_t<T, N> getNode(uint64_t offset) const override {
		btree_node_t<T, N> resultat;

		file.seek(offset, SEEK_SET);
		file.read(resultat);
		return resultat;
	}

	btree_node_t<T, N> getRootNode() const override {
		return getNode(rootOffset);
	}

	uint64_t getRootOffset() const override {
		return rootOffset;
	}

	uint64_t addNode(const btree_node_t<T, N>& node) override {
		uint64_t offset = file.seekToEndWithPadding(8);
		file.write(node);
		return offset;
	}

	void overwriteNode(uint64_t offset, const btree_node_t<T, N>& node) override {
		file.seek(offset);
		file.write(node);
	}

	// This just overwrites the initial data, currently the first three longs.  Should save time
	// if the children and elements have not been modified.
	void overwriteNodeHeader(uint64_t offset, const btree_node_t<T, N>& node) override {
		file.seek(offset);
		file.write(&node, sizeof(node.header));
	}

	off_t rootOffset;
	FdHandle file;
};

#endif //EXCESSIVE_BTREE_H
