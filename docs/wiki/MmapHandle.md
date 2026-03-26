# MmapHandle: Memory-Mapped I/O

`MmapHandle` provides efficient interaction with files by mapping them directly into the process's memory space.

## Characteristics

- **Zero-Copy I/O**: Access file data directly in memory, bypassing standard read/write system call overhead.
- **Ease of Use**: Read and write complex structures as if they were already in RAM.
- **Reference Integrity**: Maintains a reference to the parent `FdHandle` to ensure the file descriptor remains open.
- **Safe Lifecycle**: Memory is automatically unmapped (`munmap`) when the handle is destroyed.
- **Stateful Cursor**: Maintains an internal cursor for sequential read/write operations.

## Usage

### Reading and Writing Structs

```cpp
#include "fs/FdHandle.h"

struct Config {
    int id;
    char name[16];
};

void mmapExample() {
    FdHandle handle = FdHandle::open("config.dat", O_RDWR | O_CREAT, 0660);
    // Map the first 1KB of the file
    MmapHandle mmap = handle.getMmapHandle(0, 1024);
    if (!mmap) return;

    Config c{1, "Service"};
    mmap.write(c); // Advances cursor

    mmap.seek(0, SEEK_SET);
    Config readBack = mmap.read<Config>();
}
```

### Direct Memory Access

```cpp
void directAccess() {
    FdHandle handle = FdHandle::open("data.bin", O_RDWR);
    MmapHandle mmap = handle.getMmapHandle(0, 4096);

    // Get raw pointer to offset 512
    uint32_t* ptr = mmap.directPointer<uint32_t>(512);
    *ptr = 0xABCDEF01; // Direct modification
}
```

## API Reference

### Creation

`MmapHandle` instances are created via `FdHandle::getMmapHandle(offset, size, prot, flags)`.

### Operations

- **Reading**: `read<T>()`, `read(buf, size)`.
- **Writing**: `write(value)`, `write(buf, size)`.
- **Positioning**: `seek(offset, whence)`.
- **Direct Access**: `directPointer<T>(offset)`.

## Considerations

- **Mapping Size**: Operations beyond the mapped region are truncated.
- **Concurrency**: Avoid unsynchronized concurrent writes to the same memory location.
- **Persistence**: While `mmap` is efficient, use `msync` or `FdHandle::flush()` if immediate persistence is required.
