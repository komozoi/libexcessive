
# LinkedList: Doubly Linked List

`LinkedList` is a standard doubly linked list implementation with a focus on ease of use and predictable memory
management.

## Characteristics

- **Doubly Linked**: Efficient insertion and removal from both ends.
- **STL-Compatible Iterators**: Supports standard C++ iteration patterns (`for (T& item : list)`).
- **RAII-Compliant**: Automatically manages the lifecycle of its elements.

## Usage

### Basic Operations

```cpp
#include "ds/LinkedList.h"
#include <iostream>

void linkedListExample() {
    LinkedList<int> list;
    
    list.add(10);
    list.add(20);
    list.addFirst(5);

    for (int val : list) {
        std::cout << val << " "; // 5 10 20
    }

    list.removeFirst();
    int last = list.removeLast();
}
```

## API Reference

### Core Methods

- `add(value)` / `addLast(value)`: Appends an element to the end.
- `addFirst(value)`: Prepends an element.
- `remove(value)`: Removes the first occurrence of a value.
- `removeFirst()` / `removeLast()`: Removes and returns an element from either end.
- `contains(value)`: Checks if a value exists in the list.
- `size()`: Returns the number of elements.
- `clear()`: Removes all elements.

### Iterators

- `begin()` / `end()`: Standard iterators.
- `rbegin()` / `rend()`: Reverse iterators.
