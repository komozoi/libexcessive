---
title: "FreeSpaceFile: Managed File Space Allocation"
description: "A best-fit allocator for on-disk file regions, providing dynamic allocation and deallocation within a single file."
keywords: "freespacefile, allocation, on-disk, storage, fragmentation, database"
category: "Filesystem"
---

# FreeSpaceFile: Managed File Space Allocation

`FreeSpaceFile` provides a robust foundation for building custom file formats by managing free and occupied regions
within a file. It uses an on-disk B-Tree to track available space, enabling efficient "best-fit" allocation and reuse
of previously freed file regions.

## Characteristics

`FreeSpaceFile` is designed to simplify the complexity of manual file offset management. Key characteristics include:

- **Best-Fit Allocation**: The internal B-Tree is specifically sorted to return the smallest available region that can
  satisfy an allocation request, minimizing fragmentation.
- **Automatic Expansion**: If no suitable free region is found, `FreeSpaceFile` automatically expands the underlying file
  and returns space from the new end of the file.
- **B-Tree Backed**: Uses a persistent B-Tree to store and index free regions by size and offset, ensuring fast lookups
  even in very large files.
- **Header Protection**: Includes mechanisms to track the end of the metadata header, ensuring that allocations never
  overwrite the file's own structural information.
- **Flexible Ownership**: Supports both move semantics for `FdHandle` ownership and shared access via reference-counted
  `FdHandle` copies.

## Usage Example

```cpp
#include "fs/FreeSpaceFile.h"
#include "fs/FdHandle.h"
#include <fcntl.h>

void example() {
    // Open a file and wrap it in a FreeSpaceFile
    FdHandle handle = FdHandle::open("myformat.db", O_RDWR | O_CREAT, 0660);
    FreeSpaceFile fs(std::move(handle));

    // Allocate 1024 bytes
    // If the file is empty, this expands the file after the B-Tree header.
    off_t offset1 = fs.getFreeRegion(1024);

    // Later, mark that region as free
    fs.markFreeRegion(offset1, 1024);

    // A subsequent request for 512 bytes will reuse the same offset
    // because it is the smallest sufficient free region available.
    off_t offset2 = fs.getFreeRegion(512); 
    // offset2 == offset1
}
```

## API Reference: FreeSpaceFile

### `explicit FreeSpaceFile(FdHandle&& file)`

**Arguments:**

- `file`: An R-value reference to an `FdHandle`. The `FreeSpaceFile` takes ownership of this handle.

**Description:**

Constructs a `FreeSpaceFile` using the provided `FdHandle`. It initializes the internal B-Tree starting at offset 0 of
the file.

### `explicit FreeSpaceFile(const FdHandle& file)`

**Arguments:**

- `file`: A constant reference to an `FdHandle`.

**Description:**

Constructs a `FreeSpaceFile` that shares the underlying file descriptor with the provided `FdHandle`.

### `void markFreeRegion(off_t start, uint32_t length)`

**Arguments:**

- `start`: The starting byte offset of the region to be freed.
- `length`: The size of the region in bytes.

**Return Value:**

None.

**Description:**

Records a region of the file as being available for future allocations. This information is persisted in the internal
B-Tree. Note that `FreeSpaceFile` currently does not automatically coalesce adjacent free regions.

### `off_t getFreeRegion(uint32_t length)`

**Arguments:**

- `length`: The number of contiguous bytes requested.

**Return Value:**

Returns the absolute byte offset within the file where the requested space begins.

**Description:**

Searches the free region B-Tree for the smallest region that is at least `length` bytes long. If an exact or larger
match is found, that region is removed from the B-Tree and its offset is returned. If no suitable region exists, the
file is expanded, and the offset of the new space is returned.

### `off_t getHeaderEnd()`


**Return Value:**

Returns the byte offset immediately following the end of the B-Tree metadata header.

**Description:**

Provides the boundary after which user data can safely be stored without interfering with the `FreeSpaceFile`'s
internal management structures.

---

## How it Works: Technical Overview

`FreeSpaceFile` manages file space by maintaining a persistent registry of "holes" (unused segments) within the file's
structure.

### The Best-Fit B-Tree Strategy

The core of `FreeSpaceFile` is a `BTree` of `free_region_pair_t` structures, where each pair contains an `offset` and a
`size`. The efficiency of the allocation logic depends entirely on how these pairs are sorted within the tree.

The comparison function is specifically designed to facilitate a "Best-Fit" allocation strategy:
1. **Size First (Ascending)**: Regions are primarily sorted by size. This allows the `BTree::findNext` method to quickly
   locate the smallest region that is greater than or equal to the requested size.
2. **Offset Second (Ascending)**: If multiple regions have the same size, they are sorted by their offset in the file.
   This ensures that space towards the beginning of the file is reused before expanding the file further.

### Allocation and Expansion Logic

When `getFreeRegion(length)` is called, the system performs a search in the B-Tree for a pair with `size` at least 
`length`. 

- **Reuse**: If a match is found, the entry is removed from the tree. If the found region is significantly larger than
  requested, the remaining sub-region is **not** currently split and re-inserted; the entire region is consumed by the
  caller. This simplifies implementation but means callers should manage large block splitting if fragmentation becomes
  a concern.
- **Expansion**: If the B-Tree contains no sufficient regions, the system calls `seekToEndWithPadding(8)` on the 
  underlying `FdHandle`. This ensures that new allocations are aligned to 8-byte boundaries. The file is then extended 
  using `ftruncate` to reserve the requested space, and the previous end-of-file offset is returned.

### Persistence

Because the underlying `BTree` is initialized at offset 0 of the `FdHandle`, all information about free regions is
stored directly within the file being managed. This makes `FreeSpaceFile` a self-contained solution for managing
complex file-based data structures.
