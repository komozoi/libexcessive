# BTree: Persistent On-Disk Sorted Storage

`BTree` is a template-based implementation of a B-Tree that stores data directly on disk, enabling efficient persistence
and retrieval of large datasets.

## Characteristics

- **On-Disk Storage**: Data persists across application restarts, stored within a file managed by `FdHandle`.
- **Logarithmic Complexity**: Operations like `insert`, `find`, and `remove` scale efficiently with data size.
- **Low Memory Footprint**: Loads only the necessary nodes into RAM, making it suitable for datasets larger than
  available memory.
- **Type Safety**: Uses C++ templates for storing POD (Plain Old Data) structures.

## Notes

Currently `remove()` is not properly implemented.

## Usage

### Storing and Finding Data

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
    FdHandle file = FdHandle::open("store.btree", O_RDWR | O_CREAT, 0644);
    // BTree at offset 0
    BTree<Entry> tree(file, 0, Entry::compare);

    // Initial insertion
    tree.insert({42, 3.14});

    Entry search{42, 0};
    if (tree.find(search)) {
        // search.value is now 3.14
    }
}
```

### Navigating Sorted Data

```cpp
void findNextExample(BTree<Entry>& tree) {
    Entry start{10, 0};
    // Finds the entry with key >= 10
    if (tree.findNext(start)) {
        // start is updated with the next available entry
    }
}
```

## API Reference

### Initialization

`BTree<T, N> tree(fd_handle, root_offset, compare_func)`

- `T`: POD data type.
- `N`: Number of elements per node (default: 63).
- `root_offset`: File offset where the tree starts.
- `compare_func`: `int (*)(const T&, const T&)` for ordering.

### Operations

- `insert(val)`: Adds an entry if the key is not present.
- `overwrite(val)`: Adds or updates an entry.
- `find(val)`: Retrieves an entry matching the key.
- `findNext(val)`: Retrieves the first entry with a key >= input.
- `remove(val)`: Deletes an entry (Note: basic implementation, does not rebalance).
- `initialize()`: Formats the file region for a new tree.
