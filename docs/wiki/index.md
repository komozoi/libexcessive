# LibExcessive Wiki

LibExcessive is a high-performance C++ library designed for data-heavy applications. It provides RAII-based file I/O,
on-disk data structures, and efficient memory management utilities.

## Core Features

- **High-Performance I/O**: RAII-based file handles, thread-safe operations, and seamless memory mapping.
- **On-Disk Data Structures**: Persistent B-Trees for efficient storage and retrieval.
- **Fixed-Width Bigints**: Non-allocating large integer arithmetic (128, 256 bits, etc.).
- **Memory Management**: Smart pointers with Copy-on-Write (COW) and custom allocator friendliness.
- **Containers**: Simple, efficient alternatives to standard library containers.

## Documentation Index

- [**FdHandle**](FdHandle.md) - Smart file descriptor management and I/O.
- [**MmapHandle**](MmapHandle.md) - High-performance memory-mapped I/O.
- [**BTree**](BTree.md) - Persistent on-disk sorted data structures.
- [**Bigint**](Bigint.md) - Fixed-width large integer arithmetic.
- [**LinkedList**](LinkedList.md) - Doubly linked list implementation.

## Related Resources

- [Project README](../../README.md)
