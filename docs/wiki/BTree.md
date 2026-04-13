---
title: "BTree: Persistent On-Disk Storage"
description: "High-performance, persistent B-Tree implementation for on-disk sorted data storage and retrieval."
keywords: "btree, on-disk, storage, database, performance, indexing"
category: "Data Structures"
---

# BTree: Persistent On-Disk Sorted Storage

This class is useful for building on-disk mappings or sorted search structures.

`BTree` is a template-based implementation of a B-Tree that stores data directly on disk, enabling efficient persistence
and retrieval of large datasets. It uses `FdHandle` for its underlying I/O operations and maintains a shallow tree
structure to minimize disk access.

## Characteristics

`BTree` is designed for scenarios where data must persist across application restarts and may exceed available physical
memory. Its key characteristics include:

- **On-Disk Storage**: All data is stored within a file managed by `FdHandle`. Nodes are read and written to disk as
  needed.
- **Logarithmic Complexity**: Search, insertion, and deletion operations have a time complexity of O(log N), where N is
  the number of elements.
- **Low Memory Footprint**: Only the path from the root to the target leaf needs to be in memory at any given time,
  allowing for the management of datasets much larger than RAM.
- **Type Safety**: Uses C++ templates to store POD (Plain Old Data) structures, ensuring compile-time type checking.
- **Tunable Node Degree**: The number of elements per node can be adjusted via template parameters to optimize for
  different storage media and access patterns.
- **Multi-Key Sorting**: Some support for multiple keys, each item is scored and sorted internally, so the use of
  multiple keys depends on the comparison function.
- **Next-Best Search**: Supports efficient range scans by providing a `findNext` method.

## Basic Example

```cpp
#include <fs/FdHandle.h>
#include <fs/BTree.h>
#include <fcntl.h>

struct Entry {
    int key;
    double value;

    static int compare(const Entry& a, const Entry& b) {
        return a.key - b.key;
    }
};

void btreeExample() {
    // Open a file for the BTree
    FdHandle file = FdHandle::open("store.btree", O_RDWR | O_CREAT, 0644);

    // Create a BTree instance at offset 0
    BTree<Entry> tree(file, 0, Entry::compare);

    // Insert data
    tree.insert({42, 3.14});

    // Retrieve data
    Entry search{42, 0};
    if (tree.find(search)) {
        // search.value now contains 3.14
    }
}
```

## API Reference

### `BTree<T, N>(FdHandle &&file, off_t rootOffset, int(*compare)(const T &, const T &))`

**Arguments:**

- `file`: An `FdHandle` (moved or copied) representing the storage file.
- `rootOffset`: The byte offset in the file where the BTree structure begins.
- `compare`: A pointer to a function that defines the ordering of elements of type `T`. It should return `<0` if
  `a < b`, `0` if `a == b`, and `>0` if `a > b`.

**Description:**

Constructs a `BTree` object associated with a specific file and offset. `T` is the type of element stored, and `N` is
the maximum number of elements per node (degree).  This will automatically initialize the BTree if the file is new,
using the `initialize()` method; see below for more details.

### `void initialize()`

**Return Value:**

None.

**Description:**

Formats the file region at `rootOffset` to initialize a new, empty BTree. This involves creating the initial root node
header and marking it as a leaf.

This method usually does not need to be called manually, unless the file is being reinitialized, in which case all
previous data in the structure will be lost.  The BTree class automatically calls this method during construction if
the file is new, using the `FdHandle::isNew()` method.

### `bool insert(const T &val)`

**Arguments:**

- `val`: The element to insert.

**Return Value:**

Returns `true` if the element was successfully inserted. Returns `false` if an element with an equivalent key (according
to the comparison function) already exists in the tree.

**Description:**

Inserts a new element into the BTree.

### `bool overwrite(const T &val)`

**Arguments:**

- `val`: The element to insert or update.

**Return Value:**

Returns `true` if the element was inserted or updated.

**Description:**

Inserts `val` into the tree. If an element with an equivalent key already exists, it is overwritten with the new data.

### `bool find(T &val)`

**Arguments:**

- `val`: An element containing the key to search for.

**Return Value:**

Returns `true` if a matching element was found. In this case, the input `val` is updated with the full data stored in
the tree. Returns `false` otherwise.

**Description:**

Performs a search for an element equivalent to `val`.

### `bool findNext(T &val)`

**Arguments:**

- `val`: An element containing a lower-bound key.

**Return Value:**

Returns `true` if an element was found that is greater than or equal to `val`. The input `val` is updated with the found
element. Returns `false` if no such element exists (i.e., `val` is greater than the largest element in the tree).

**Description:**

Finds the first element in the tree whose key is not less than the key in `val`. This is useful for range scans.

### `bool remove(T &val)`

**Arguments:**

- `val`: An element containing the key to remove.

**Return Value:**
Returns `true` if the element was found and removed. Returns `false` otherwise.

**Description:**

Removes the element matching the key in `val`.

**NOTE:** The current implementation of `remove` is basic and may not perform full rebalancing of the tree in all edge
cases.  Use this method with caution.

### `off_t getHeaderEndOffset() const`

**Return Value:**

Returns the file offset immediately following the BTree's root header.

**Description:**

Useful for storing metadata immediately after the BTree root structure in a file.  This allows the construction of
complex on-disk data structures, which may include multiple trees or other complex on-disk structures.

### `uint64_t getRootOffset() const`

**Return Value:**

Returns the absolute file offset of the root node.

**Description:**

Retrieves the location of the root node, as set in the constructor.

---

## How it Works: Technical Overview

The `BTree` implementation is a high-performance, disk-resident data structure designed to handle large datasets with
minimal memory overhead.

### Node Structure and Page Management

The tree is composed of nodes (`btree_node_t`), which are fixed-size blocks of memory that correspond directly to
regions in the underlying file. Each node contains a header (`btree_node_header_t`) that stores the current count of
elements and other metadata.

Internal nodes contain `k` elements and `k+1` child pointers (file offsets). Leaf nodes contain only elements. The
maximum number of elements per node is defined by the template parameter `N`. By choosing a large `N`, the tree
maintains a very high fan-out, resulting in a very shallow tree (often only 3-4 levels deep even for millions of
entries), which minimizes the number of disk reads required for any operation.

### Search and Traversal

Traversal begins at the root offset provided during construction. For each node, the library performs a binary search
to find the first element that is not less than the search key. If a match is found,
the operation completes. If not, and the node is an internal node, the library follows the child pointer corresponding
to the search path and repeats the process. `FdHandle` is used to read nodes from disk into temporary memory buffers
during traversal.

### Insertion and Tree Growth

Insertion follows a "proactive split" strategy to maintain the B-Tree invariants. As the library traverses down the tree
to find the insertion point, it checks if any node on the path is full (contains `N` elements). If a full node is
encountered, it is split into two nodes, and the median element is promoted to the parent.

If the root node itself is full and needs to be split, a new root node is created, and the height of the tree increases.
This ensures that a leaf node always has room for at least one more element when the insertion point is reached,
avoiding the need for complex multi-pass or recursive split logic.

The root node never moves; when a new root node is created, the old root is moved to a new location in the file, and
the new root is written at the original root offset.

### Persistence and Concurrency

All modifications to nodes are immediately written back to disk using `FdHandle`. This ensures that the tree structure
remains consistent in the file. While the `BTree` class itself does not implement internal locking for multi-threaded
access, it relies on the thread-safe nature of `FdHandle` for its I/O. For concurrent access to the tree structure
itself, external synchronization (e.g., a mutex or read-write lock) is required to prevent race conditions during splits
and updates.

In the future, `pread` and mmap usage will allow better concurrency.  This structure would then allow safe concurrent
reads, concurrent reads and writes for non-BTree data in the file, but not allow safe concurrent writes due to the
nature of the tree structure.
