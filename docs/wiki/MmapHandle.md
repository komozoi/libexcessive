# MmapHandle: Memory-Mapped I/O

`MmapHandle` provides efficient interaction with files by mapping them directly into the process's memory space. This
bypasses the overhead of standard read/write system calls and allows for direct memory access to file contents.

## Characteristics

`MmapHandle` is optimized for high-performance data access and manipulation. Its key characteristics include:

- **Zero-Copy I/O**: Enables access to file data directly in process memory, bypassing the standard kernel-to-user-space
  copying required by `read` and `write` system calls.
- **Structured Data Access**: Allows reading and writing complex C++ structures as if they were already residing in RAM.
- **Reference Integrity**: Automatically increments the reference count of the parent `FdHandleData`, ensuring the file
  descriptor remains open and valid for the duration of the mapping.
- **RAII Lifecycle**: Memory is automatically unmapped via `munmap(2)` and the associated file handle reference is
  released when the `MmapHandle` instance is destroyed.
- **Stateful Cursor**: Maintains an internal cursor to support sequential read and write operations, mimicking the
  behavior of a standard file pointer.

## Usage Examples

### Direct Memory Access

```cpp
void directAccess() {
    FdHandle handle = FdHandle::open("data.bin", O_RDWR);
    MmapHandle mmap = handle.getMmapHandle(0, 4096);

    // Get raw pointer to offset 512
    uint32_t* ptr = mmap.directPointer<uint32_t>(512);
    *ptr = 0xABCDEF01; // Direct modification in memory
}
```

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

    // Structs can be written directly to memory:
    Config* ptr = mmap.directPointer<Config>(0);
    ptr[5] = {42, "Direct"};

    // If you prefer to use a cursor, like traditional file I/O:
    Config c{1, "Cursor"};
    mmap.write(c); // Advances cursor

    // Unlike with raw FdHandle:seek, MmapHandle::seek is thread-safe even when concurrently writing to the file:
    mmap.seek(0, SEEK_SET);
    Config readBack = mmap.read<Config>();
}
```

## API Reference

### `operator bool() const`

Returns `true` if the handle currently holds a valid memory mapping, and `false` otherwise.

### `template<typename T> T* directPointer(off_t where = 0)`

**Arguments:**

- `where`: Offset from the start of the mapping.

**Return Value:**

Returns a pointer of type `T*` to the memory address at the specified offset.  This offset is relative to the start of
the mapping, **not the start of the file**.

**Description:**

Provides direct pointer access to the mapped file content, allowing for high-performance in-place operations.

### `ssize_t write(const void* value, size_t size)`

**Arguments:**

- `value`: Pointer to the source data buffer.
- `size`: Number of bytes to write.

**Return Value:**

Returns the number of bytes actually written. If the write would exceed the bounds of the mapping, it is truncated.

**Description:**

Copies `size` bytes from `value` to the memory region at the current cursor position and advances the cursor.

### `template<class T> ssize_t write(const T& value)`

**Arguments:**

- `value`: Constant reference to an object of type `T`.

**Return Value:**

Returns the number of bytes written.

**Description:**

A template convenience method for writing a single POD object.

### `ssize_t read(void* value, size_t size)`

**Arguments:**

- `value`: Pointer to the target buffer.
- `size`: Number of bytes to read.

**Return Value:**

Returns the number of bytes actually read. Reads are truncated if they would exceed the bounds of the mapping.

**Description:**

Copies `size` bytes from the current cursor position in the mapping to `value` and advances the cursor.

### `template<class T> ssize_t read(T& value)`

**Arguments:**

- `value`: Reference to an object of type `T`.

**Return Value:**

Returns the number of bytes read.

**Description:**

A template convenience method for reading data into a single POD object.

### `template<class T> T read()`

A convenience template method that reads and returns a single object of type `T` from the current cursor position.

### `off_t seek(off_t where, int whence = SEEK_SET)`

**Arguments:**

- `where`: The target offset relative to `whence`.
- `whence`: The reference point (`SEEK_SET`, `SEEK_CUR`, or `SEEK_END`).

**Return Value:**

Returns the new cursor position relative to the start of the mapping, or -1 on error.

**Description:**

Adjusts the internal cursor position within the mapped memory region. The cursor is clamped to the range
`[0, mapping_size]`.

---

## How it Works: Technical Overview

The `MmapHandle` class is a high-level abstraction over the `mmap(2)` system call, designed to integrate seamlessly with
the `FdHandle` resource management system.

### Mapping and Resource Management

A `MmapHandle` is typically created via `FdHandle::getMmapHandle`. During construction, the library invokes `mmap(2)`
with the underlying file descriptor, the requested offset (which must be page-aligned in the kernel, though the library
handles some of this complexity), and the requested size.

Crucially, upon successful mapping, the `MmapHandle` increments the reference count of the internal `FdHandleData`
object. This ensures that the file descriptor remains open in the kernel's file table even if all `FdHandle` instances
that previously referenced it are destroyed. The lifecycle of the mapping is tied to the `MmapHandle` instance; its
destructor invokes `munmap(2)` and then decrements the `FdHandleData` reference count, potentially closing the file
descriptor if no other references remain.

### Cursor and I/O Emulation

While `mmap` provides direct memory access, `MmapHandle` also provides a stream-like interface through its `read`,
`write`, and `seek` methods. This is implemented using an internal `cursor` pointer which is initialized to the start of
the mapped memory block.

- **Sequential I/O**: The `read` and `write` methods use `memcpy` to transfer data between the process's heap/stack and
  the mapped memory. After the copy, the `cursor` is incremented by the number of bytes transferred. Bound checking is
  performed against the `end` pointer of the mapping to prevent segmentation faults.
- **Positioning**: The `seek` method allows for relative and absolute positioning of the `cursor`, similar to how
  `lseek(2)` works for file descriptors, but operating entirely within the process's address space.

### Memory Consistency and Persistence

Because `MmapHandle` uses `MAP_SHARED` by default, changes made to the mapped memory are eventually propagated to the
underlying file by the operating system's virtual memory manager. However, this propagation is asynchronous. For
applications requiring guaranteed durability, `FdHandle::flush()` or the system's `msync(2)` call should be used to
force synchronization of the memory-mapped pages to the physical storage medium.
