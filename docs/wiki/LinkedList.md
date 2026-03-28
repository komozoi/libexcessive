
---
title: "LinkedList: Efficient Doubly Linked List"
description: "A professional doubly linked list implementation with simple, clean APIs and no external dependencies."
keywords: "linkedlist, data structure, performance, c++"
category: "Data Structures"
---

# LinkedList: Doubly Linked List

`LinkedList` is a standard template-based implementation of a doubly-linked list. It provides efficient insertion and
deletion at both ends and supports bi-directional iteration.

## Characteristics

`LinkedList` is suitable for collections where frequent insertions and removals are expected, especially at the
beginning or end of the sequence.

- **Dynamic Memory Allocation**: Each element is stored in a separate node (`linkedlist_value_container_t`) allocated on
  the heap.
- **Bi-directional Traversal**: Each node contains pointers to both the next and previous elements.
- **Efficient Modifications**: Adding or removing elements at the head or tail is an O(1) operation.
- **Stateful Cursor**: Includes an internal cursor for efficient sequential access without manually managing iterators.
- **STL-Compatible Iterators**: Provides `iterator`, `const_iterator`, `reverse_iterator`, and `const_reverse_iterator`
  for compatibility with C++ standard library algorithms.

## Usage Example

```cpp
#include "ds/LinkedList.h"
#include <iostream>

void listExample() {
    LinkedList<int> list;

    // Adding elements
    list.add(10);
    list.add(20);
    list.insertAtBeginning(5);

    // Iterating with standard loops
    for (int value : list) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Removing elements
    list.remove(1); // Removes element at index 1
}
```

## API Reference

### Constructors and Destructor

- `LinkedList<T>()`
    - Constructs an empty list.
- `LinkedList<T>(const T *src, int size)`
    - Constructs a list and populates it with `size` elements from the provided array `src`.
- `~LinkedList()`
    - Destroys the list and deallocates all nodes and their stored values.

### Insertion and Removal

- `void add(T item)`
    - Appends `item` to the end of the list.
- `void insertAtBeginning(T item)`
    - Prepends `item` to the beginning of the list.
- `void addMany(const T *items, int count)`
    - Appends `count` elements from the `items` array to the end of the list.
- `T pop()`
    - Removes and returns the last element in the list.
- `T remove(int i)`
    - Removes and returns the element at index `i`. This is an O(N) operation.
- `bool removeByElement(T element)`
    - Searches for the first occurrence of `element` and removes it. Returns `true` if found and removed.
- `void clear()`
    - Removes all elements from the list.

### Access and Cursor Management

- `int size() const`
    - Returns the number of elements currently in the list.
- `T &get(int i) const`
    - Returns a reference to the element at index `i`. O(N) complexity.
- `void resetCursor()`
    - Resets the internal cursor to the beginning of the list.
- `void cursorToLast()`
    - Moves the internal cursor to the last element.
- `bool isCursorValid()`
    - Returns `true` if the cursor currently points to a valid node.
- `T &next()`
    - Returns the element at the current cursor position and advances the cursor to the next node.
- `T &getCursor()`
    - Returns a reference to the element at the current cursor position.
- `void setCursor(int i)`
    - Moves the internal cursor to the node at index `i`.

### Iterators

The class provides standard `begin()`, `end()`, `cbegin()`, `cend()`, `rbegin()`, and `rend()` methods returning
appropriate iterator types.

---

## Usage Notes

`LinkedList` manages a sequence of nodes, where each node is an instance of `linkedlist_value_container_t`.

### Index-Based Access

While `LinkedList` supports index-based access via `get(int i)` and `remove(int i)`, these operations are O(N) as they
require traversing the list from either the head or the tail (depending on which is closer) to reach the target index.
For heavy index-based workloads, `ArrayList` would be a more appropriate choice.
