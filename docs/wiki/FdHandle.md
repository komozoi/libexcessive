# FdHandle: Smart File Descriptor Management

`FdHandle` provides a thread-safe, reference-counted wrapper around a standard file descriptor, with RAII semantics.

## Characteristics

- **RAII Lifecycle**: Automatically closes the file descriptor when the last `FdHandle` reference is destroyed.
- **Reference Counting**: Multiple instances can share the same underlying file descriptor, preventing premature
  closure.
- **Thread-Safety**: Concurrent read and write operations are supported.
- **State Tracking**: The `isNew()` method identifies if the file was created or already existed.
- **Mmap Integration**: Seamlessly creates `MmapHandle` instances for high-performance memory-mapped I/O.

## Usage

### Basic Reading and Writing

```cpp
#include "fs/FdHandle.h"
#include <fcntl.h>

void example() {
    // Open for writing, create if it doesn't exist
    FdHandle handle = FdHandle::open("data.bin", O_RDWR | O_CREAT, 0660);
    if (!handle) return;

    // Template methods handle size automatically for POD types
    uint64_t value = 0x12345678;
    handle.write(value);

    handle.seek(0, SEEK_SET);
    uint64_t readValue;
    handle.read(readValue);
}
```

### Reference Counting

```cpp
void sharing() {
    FdHandle h1 = FdHandle::open("test.txt", O_RDONLY);
    {
        FdHandle h2 = h1; // Both share the same FD
        // h1.numReferences() == 2
    }
    // h2 is gone, but the FD remains open because of h1
}
```

## API Reference

### Opening and Creation

| Method                                   | Description                             |
|------------------------------------------|-----------------------------------------|
| `static FdHandle open(path, mode)`       | Opens a file with standard `O_*` flags. |
| `static FdHandle open(path, mode, flag)` | Opens with flags and file permissions.  |
| `static FdHandle from(fd)`               | Wraps an existing file descriptor.      |
| `static void pipe(reader, writer)`       | Creates a Unix pipe.                    |

### Operations

- **I/O**: `read(buf, size)`, `write(buf, size)`, and template variants `read<T>()`, `write<T>(val)`.
- **Positioning**: `seek(offset, whence)`, `seekToEndWithPadding(alignment)`.
- **Management**: `flush()`, `close()`, `markToClose()`.
- **Metadata**: `isNew()`, `numReferences()`, `getFd()`.

---

## FdTransaction

`FdTransaction` is a helper for scoped file operations, ensuring the underlying handle remains valid and providing a
convenient threadsafe interface for seeking and sequential I/O.

### Example

```cpp
void process(const FdHandle& handle) {
    FdTransaction tx(handle);
    tx.seek(1024);
    auto data = tx.read<MyStruct>();
    tx.seek(1024);
    tx.write(data);
}
```
