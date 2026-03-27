# FdHandle: Smart File Descriptor Management

`FdHandle` provides a thread-safe, reference-counted wrapper around a standard file descriptor, with RAII semantics.
It simplifies file I/O by managing the lifecycle of file descriptors and providing a more robust interface than raw
system calls.

## Characteristics

`FdHandle` is designed for high-performance, robust file I/O. Its primary characteristics include:

- **RAII Lifecycle**: Automatically closes the underlying file descriptor when the last `FdHandle` reference is
  destroyed, preventing resource leaks.
- **Reference Counting**: Multiple `FdHandle` instances can share the same underlying file descriptor, allowing safe
  sharing across different parts of an application.
- **Thread-Safety**: Concurrent read and write operations are supported through internal locking, protecting the file
  offset and internal state.
- **State Tracking**: Includes mechanisms to identify if a file was newly created or already existed during the `open`
  call.
- **Mmap Integration**: Provides a seamless interface for creating `MmapHandle` instances for high-performance
  memory-mapped I/O.
- **Queued Writes**: Supports merging and delaying writes for optimized I/O performance, reducing system call overhead.

## Usage Example

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

## API Reference: FdHandle

### `static FdHandle open(const char* path, int mode)`

**Arguments:**

- `path`: A C-string representing the filesystem path to the file.
- `mode`: Bitwise OR of flags (e.g., `O_RDONLY`, `O_WRONLY`, `O_RDWR`, `O_CREAT`).

**Return Value:**

Returns an `FdHandle` instance. If the operation fails, the returned handle's boolean conversion will evaluate to
`false`.

**Description:**

Opens the file specified by `path`. If the file is successfully opened, it is wrapped in an `FdHandle` which manages its
lifecycle through reference counting.

### `static FdHandle open(const char* path, int mode, int flag)`

**Arguments:**

- `path`: A C-string representing the filesystem path to the file.
- `mode`: Bitwise OR of flags (e.g., `O_RDONLY`, `O_WRONLY`, `O_RDWR`, `O_CREAT`).
- `flag`: File permissions (e.g., `0660`) used when `O_CREAT` is specified.

**Return Value:**

Returns an `FdHandle` instance. If the operation fails, the returned handle's boolean conversion will evaluate to
`false`.

**Description:**

Similar to the two-argument `open` method, but allows specifying permissions for a newly created file.

### `static FdHandle from(int fd)`

**Arguments:**

- `fd`: A raw file descriptor.

**Return Value:**

Returns an `FdHandle` wrapping the provided file descriptor.

**Description:**

Allows wrapping an existing file descriptor into the `FdHandle` system. The system will then take over management and
reference counting of that descriptor.  This is particularly useful for sockets, which as `FdHandle` instances, can be
passed around and used the same way as regular files.

### `static void pipe(FdHandle& reader, FdHandle& writer)`

**Arguments:**

- `reader`: Reference to an `FdHandle` that will receive the read end of the pipe.
- `writer`: Reference to an `FdHandle` that will receive the write end of the pipe.

**Return Value:**

None.

**Description:**

Creates a Unix pipe and assigns the read and write ends to the provided `FdHandle` instances.

### `ssize_t write(const void* value, size_t size) const`

**Arguments:**

- `value`: A pointer to the buffer containing data to be written.
- `size`: The number of bytes to write.

**Return Value:**

Returns the number of bytes written, or a negative value on error (corresponding to standard `write` return values).

**Description:**

Writes `size` bytes from the buffer pointed to by `value` to the file. This operation is thread-safe and updates the
file offset accordingly.

### `template<class T> ssize_t write(const T& value) const`

**Arguments:**

- `value`: A constant reference to an object of type `T`.

**Return Value:**

Returns the number of bytes written, or a negative value on error.

**Description:**

A template convenience method that writes the memory representation of `value` to the file. This is intended for POD
(Plain Old Data) types.

### `void queueWrite(const void* value, size_t size, off_t where) const`

**Arguments:**

- `value`: A pointer to the buffer containing data to be written.
- `size`: The number of bytes to write.
- `where`: The file offset at which the write should occur.

**Return Value:**

None.

**Description:**

Queues a write operation at the specified offset. If the write is adjacent to an existing queued write, they may be
merged into a single larger write operation to optimize performance.

### `template<class T> void queueWrite(const T& value, off_t where) const`

**Arguments:**

- `value`: A constant reference to an object of type `T`.
- `where`: The file offset at which the write should occur.

**Return Value:**

None.

**Description:**

A template convenience method that queues the memory representation of `value` at the specified offset.

### `template<class T> void queueWriteWithPadding(const T& value, off_t where, uint16_t alignment) const`

**Arguments:**

- `value`: A constant reference to an object of type `T`.
- `where`: The file offset at which the write should occur.
- `alignment`: The required alignment (must be a power of two).

**Return Value:**

None.

**Description:**

Queues a write of `value` at the specified offset, but pads the written data so that the total size matches the
specified alignment.

### `ssize_t read(void* value, size_t size) const`

**Arguments:**

- `value`: A pointer to the buffer where the read data will be stored.
- `size`: The number of bytes to read.

**Return Value:**

Returns the number of bytes actually read, or a negative value on error.

**Description:**

Reads `size` bytes from the file into the buffer pointed to by `value`. This operation is thread-safe and updates the
file offset.

### `template<class T> ssize_t read(T& value) const`

**Arguments:**

- `value`: A reference to an object of type `T`.

**Return Value:**

Returns the number of bytes actually read, or a negative value on error.

**Description:**

A template convenience method that reads enough bytes to fill the memory representation of `value`.

### `bool waitForRead() const`

**Arguments:**
None.

**Return Value:**

Returns `true` if data is available for reading, or `false` on error or timeout.

**Description:**

Blocks the current thread until data becomes available on the file descriptor. This is particularly useful for
stream-like descriptors (e.g., pipes, sockets). It uses the `poll(2)` system call internally.

### `off_t seek(off_t where, int whence = SEEK_SET) const`

**Arguments:**

- `where`: The target offset.
- `whence`: The reference point for the offset (e.g., `SEEK_SET`, `SEEK_CUR`, `SEEK_END`).

**Return Value:**

Returns the resulting offset from the beginning of the file, or -1 on error.

**Description:**

Sets the file's current offset, similar to the `lseek(2)` system call.

### `off_t seekToEndWithPadding(uint8_t paddingBytes) const`

**Arguments:**

- `paddingBytes`: The alignment boundary (e.g., 16).

**Return Value:**

Returns the new file offset.

**Description:**

Seeks to the end of the file and then moves the offset forward to the next multiple of `paddingBytes`. This is useful
for maintaining alignment when appending data.

### `void flush() const`

**Arguments:**

None.

**Return Value:**

None.

**Description:**

Executes all queued write operations and ensures that they have been submitted to the operating system, then
syncs the file to disk.  After that, all previous writes and queued writes are durable.

### `void close()`

**Arguments:**

None.

**Return Value:**

None.

**Description:**

Manually closes the underlying file descriptor and invalidates the `FdHandle`. This should be used when an immediate
close is required, rather than waiting for all references to be destroyed.

### `void markToClose() const`

**Arguments:**

None.

**Return Value:**

None.

**Description:**

Marks the handle such that the file descriptor will be closed as soon as the last `FdHandle` reference is destroyed.
This can be used to signal that no further operations should be initiated.

### `bool shouldClose() const`

**Arguments:**

None.

**Return Value:**

Returns `true` if the handle has been marked to close.

**Description:**

Checks if the handle has been marked to be closed.

### `bool isNew() const`

**Arguments:**

None.

**Return Value:**

Returns `true` if the file was newly created during the call to `open`.

**Description:**

Provides feedback on whether the file was already present on disk or was created as a result of the `O_CREAT` flag.

### `int getFd() const`

**Arguments:**
None.

**Return Value:**

Returns the raw integer file descriptor.

**Description:**

Provides access to the underlying file descriptor for integration with other libraries or system calls not directly
supported by `FdHandle`.

### `std::lock_guard<std::mutex> getLock() const`

**Arguments:**
None.

**Return Value:**

A `std::lock_guard` that locks the handle's internal mutex.

**Description:**

Allows the user to manually synchronize multiple operations on the `FdHandle` by acquiring its internal lock.

### `MmapHandle getMmapHandle(off_t offset, size_t size, int prot = PROT_READ | PROT_WRITE, int flags = MAP_SHARED)`

**Arguments:**

- `offset`: The starting offset in the file to map.
- `size`: The size of the memory region to map.
- `prot`: Memory protection flags (e.g., `PROT_READ`, `PROT_WRITE`).
- `flags`: Mapping flags (e.g., `MAP_SHARED`, `MAP_PRIVATE`).

**Return Value:**

Returns an `MmapHandle` instance for the mapped region.

**Description:**

Creates a memory-mapped view of a portion of the file. The resulting `MmapHandle` will hold a reference to the
`FdHandle`, ensuring the file remains open as long as the mapping is active.

### `int numReferences() const`

**Arguments:**

None.

**Return Value:**

Returns the number of active `FdHandle` instances sharing the same file descriptor.

**Description:**

Returns the current reference count of the underlying `FdHandleData`.

---

## API Reference: FdTransaction

`FdTransaction` provides a scoped, thread-safe interface for performing multiple operations on an `FdHandle` while
ensuring the offset is correctly managed and the handle remains valid.

### `explicit FdTransaction(const FdHandle& handle)`

**Arguments:**

- `handle`: The `FdHandle` instance to operate on.

**Return Value:**

None.

**Description:**

Constructs an `FdTransaction`. This constructor acquires the internal mutex of the `FdHandle`, ensuring exclusive access
to the file descriptor until the transaction is destroyed.

### `ssize_t write(const void* value, size_t size) const`

**Arguments:**

- `value`: Pointer to the data buffer.
- `size`: Number of bytes to write.

**Return Value:**

Number of bytes written or a negative error code.

**Description:**

Writes data at the transaction's current offset.

### `ssize_t read(void* value, size_t size) const`

**Arguments:**

- `value`: Pointer to the target buffer.
- `size`: Number of bytes to read.

**Return Value:**

Number of bytes read or a negative error code.

**Description:**

Reads data from the transaction's current offset.

### `template<class T> T read() const`

**Arguments:**

None.

**Return Value:**

Returns an object of type `T`.

**Description:**

A template method that reads and returns a single object of type `T` from the transaction's current offset.

### `off_t seek(off_t where, int whence = SEEK_SET) const`

**Arguments:**

- `where`: Target offset.
- `whence`: Reference point.

**Return Value:**

The resulting absolute offset.

**Description:**

Sets the offset for the transaction's subsequent operations.

### `off_t seekToEndWithPadding(uint8_t paddingBytes) const`

**Arguments:**

- `paddingBytes`: Alignment boundary.

**Return Value:**

The resulting offset.

**Description:**

Seeks to the end of the file with the specified alignment padding.

### `bool isFile() const`

**Arguments:**

None.

**Return Value:**

`true` if the handle refers to a regular file.

**Description:**

Queries the filesystem to determine if the descriptor refers to a standard file.

### `bool isStream() const`

**Arguments:**

None.

**Return Value:**

`true` if the handle refers to a stream-like object (e.g., a pipe).

**Description:**

Queries the filesystem to determine if the descriptor refers to a stream.

---

## How it Works: Technical Overview

The `FdHandle` implementation is centered around a decoupled architecture that separates the public-facing handle from
the underlying file resources and their management.

### The Handle Registry and Lifecycle Management

Each `FdHandle` instance is a small, value-type object containing only a single 2-byte file descriptor. The actual
state associated with this descriptor is stored in a private global registry, specifically an instance of
`FdHandleState`. This registry maintains a mapping between raw file descriptors and `FdHandleData` objects.

When a file is opened or wrapped, the system checks the registry. If the descriptor is not already present, a new
`FdHandleData` instance is allocated (often using a specialized `SlabAllocator` for performance) and added to the
mapping. This `FdHandleData` object contains the actual reference count, the mutex for thread synchronization, and any
state related to queued operations.

The reference counting mechanism uses standard increment and decrement logic. When an `FdHandle` is copied, it
increments the count in the corresponding `FdHandleData`. When the `FdHandle` destructor is called, it decrements the
count. If the count reaches zero, the system automatically removes the entry from the registry, closes the underlying OS
file descriptor, and deallocates the `FdHandleData` object.

### Thread Synchronization and Atomicity

Concurrency is handled at the `FdHandleData` level. Every I/O operation (read, write, seek, etc.) must first acquire the
mutex stored within the `FdHandleData`. This ensures that even if multiple threads hold their own copies of an
`FdHandle` pointing to the same file descriptor, their operations will be serialized. This prevents interleaving of data
during writes and ensures that the internal OS file pointer remains consistent during a single logical operation.

`FdTransaction` builds upon this by holding the mutex for its entire lifetime. By acquiring the lock in the constructor
and releasing it in the destructor, `FdTransaction` provides a mechanism for users to group multiple operations
together, ensuring they appear atomic relative to other threads using the same `FdHandle`.

### Queued Write Optimization

To mitigate the performance costs of frequent, small `write(2)` system calls, `FdHandle` implements a write-queuing
system. When `queueWrite` is invoked, the data is not immediately passed to the operating system. Instead, it is stored
in a queue associated with the `FdHandleData`.

The system attempts to merge adjacent write operations. If a new write is requested at an offset that is immediately
contiguous with a previously queued write, the buffers are merged into a single, larger memory block using `realloc`.
This significantly reduces context switching and kernel overhead. These writes are eventually submitted when `flush()`
is called, when the handle is closed, or when certain internal thresholds are met.

### Memory-Mapped Integration

The `MmapHandle` integration relies on the same reference counting system. When `getMmapHandle` is called, it creates a
memory mapping using the `mmap(2)` system call. The resulting `MmapHandle` stores a reference to the same `FdHandleData`
that the `FdHandle` uses. This ensures that the file descriptor remains open and valid in the kernel as long as the
memory mapping exists, even if all `FdHandle` instances are destroyed. The mapping is released via `munmap(2)` in the
`MmapHandle` destructor.
