
# Release Notes - v0.3.1

This release is focused on bugfixes and stability improvements.  Better multithreading, less
IO contention, and more stability.

Additionally, we now officially support ARM, and FrozenBTree has been introduced to provide immutable
on-disk indexes using Mmap.

## New Features

- **FrozenBTree**:
  - Introduced `FrozenBTree` for high-performance, read-only B-Tree segments.
  - Optimized for static data with minimal memory overhead.
- **ARM Support**:
  - Added initial support and testing for ARM architectures.  Goal is to support MacOS *soon* :tm:.

## Improvements

- **B-Tree concurrency and critical logic**:
  - Fixed severe logic errors in `findNext`, `findNearest`, and `scanNode` affecting navigation and ascending order consistency.
  - Added `std::shared_mutex` to `BTreeBase` for enabling concurrent reader access while maintaining exclusive writer access.
  - Corrected node split/insertion logic in `insertNonFull`.
  - Now uses `pwrite` and `pread` to avoid conflicts with other threads accessing the same file.  Improves multithreaded performance.
- **FdHandle positional I/O**:
  - Added thread-safe positional `pread` and `pwrite` methods for concurrent I/O operations without modifying the seek cursor.
  - Replaced `FdHandleData::mutex` with `std::recursive_mutex` to eliminate deadlocks in complex transactions.
- **Concurrency & Stability**:
  - Fixed `FreeSpaceFile` SEGV/double-lock crashes under heavy `ThreadPool` contention.
- **Build System**:
  - Improved `CMakeLists.txt` with corrected recursive globbing logic.
  - Moved `testmain` out of the library target to ensure it is not included in dependent builds.
  - Suppressed spurious infinite recursion warnings in certain compiler versions.
- **Other**:
  - Added `sp<T>::numReferences()` to simplify reference count checks.
  - Added copy assignment operator to `FdHandle`.
  - Enhanced MmapHandle/FdHandle interaction.

## Bug Fixes

- **Bigint Division**:
  - Fixed a severe bug in `_internalBigintFastDiv` where dividing by 1 (or other small values) would corrupt the quotient by zeroing out intermediate limbs.
- **B-Tree Repairs**:
  - Fixed multiple compilation errors in `BTree.h` related to missing forward declarations and incorrect inheritance.
  - Resolved illegal local reference returns in `BTreeIterator`.

## Testing

- **Bigint**:
  - Added regression tests for the `bigint` division-by-one bug in `UnsignedFixedWidthBigIntTest`.
- **B-Tree**:
  - Added comprehensive regression tests for BTree concurrency, including a new `BTreeFindNext` suite.
- **FdHandle**:
  - Added 9 new concurrency/safety tests for `FdHandle::pread`/`pwrite` in `FdHandleTest` suite, covering cross-thread safety, positional consistency, and queueWrite interaction.
- **FreeSpaceFile**:
  - Added regression test for `FreeSpaceFile` concurrent allocation and free.

# Release Notes - v0.3.0

## New Features

- **Persistent Indexing and Storage**:
  - Added `Bytestring` class for optimized byte sequence management.
  - Added `DiskBytestringSearchTree` for on-disk B-Tree search with `Bytestring` keys.
- **ThreadPool**:
  - Added `ThreadPool` class for managing a pool of worker threads.
  - This class simplifies the creation and management of worker threads,
    allowing for efficient parallel execution of tasks.
- **Container Enhancements**:
  - `Queue` now inherits from the `Container` hierarchy, improving consistency across data structures.
  - Added `addMany` method to `ArrayList` and `LinkedList` to support adding elements from any generic container or iterator-compatible object.
- **Smart Pointer Improvements (`sp<T>`)**:
  - Refactored internal control blocks to use virtual methods, ensuring correct behavior for polymorphic types (e.g., `sp<Base>` managing a `Derived` instance).
  - Fixed `mut()` method to work correctly when a smart pointer has been cast to a base type.
  - Improved constructor logic using `std::enable_if` to prevent perfect-forwarding from accidentally overriding copy/move constructors.
  - Added `copy()` method to create an immediate deep copy of the managed object with a specified pointer type.

## Improvements

- **FdHandle enhancements**:
  - Added `printf` and `readLine` methods to `FdHandle` for easier string formatting and stream parsing.
  - Improved thread-safety in `FdHandle` by using `std::mutex` and `std::atomic` for reference counting.
  - Refined `close()` behavior to use reference counting instead of immediate deletion.
- **ArrayList Memory Management**:
  - Improved handling of non-trivial types during reallocations and container operations.
  - Added optimized `find` and `contains` methods.
- **LinkedList Features**:
  - Added full support for copy and move constructors and assignment operators.
- **Hash quality improvements**:
  - Improved bit mixing in `obviousHashFunction` for strings, specifically addressing quality issues for strings with lengths not divisible by 8.
- **MmapHandle Flexibility**:
  - Added default constructor and move assignment operator to `MmapHandle`.
- **HashMap Performance**:
  - Improved `insert` method to use `std::forward` for more efficient value forwarding.
- **Enhanced Error Handling**:
  - Integrated out-of-bounds exceptions into `ArrayList` and `LinkedList` for better reliability.

## Bug Fixes

- **Socket IO Hang**:
  - Fixed an infinite busy-loop in `SocketHandleData::read` that occurred when a pipe or socket was closed by the peer (`POLLHUP`).
  - This issue caused some applications to hang indefinitely when reading from a closed pipe.
- **FdHandle write merging fix**:
  - Fixed a bug in `FdHandle` where merging pending writes could result in incorrect memory moves, corrupting the write buffer.
- **sp<T> construction fix**:
  - Fixed a regression where `sp<T>` would sometimes attempt to construct the managed object from the smart pointer itself during copy operations.
- **String Formatting**:
  - Fixed several small bugs related to string formatting.

## Testing
- Added comprehensive edge-case tests for `FdHandle` concurrency and write merging.
- Added new tests for `ArrayList` and `LinkedList` memory management and generic container integration.
- Expanded `sp<T>` test suite to cover polymorphic conversions and "collapsing" constructor scenarios.
- Added leak detection tests for `HashMap`.
- Added extensive test coverage for `Bytestring` and `DiskBytestringSearchTree`.

# Release Notes - v0.2.1

This release was focused on ease of use and API improvements.  Some features were added but this was primarily
to support API changes and upgrades.  There should be no breaking changes; only added conveniences and
addition of features that should have been previously available.

## New Features

- **Polymorphic Smart Pointers**:
    - Added templated move/copy constructors and assignment operators to `sp<T>`.
    - Allows seamless conversion from `sp<Derived>` to `sp<Base>` while maintaining correct reference counting and ownership semantics.
- **Enhanced C-String Support in Hash Containers**:
    - Implemented specializations for `const char*` and `char*` keys in `HashMap` and `HashSet`.
    - These containers now correctly hash and compare string content instead of pointer addresses.
- **Improved Hash Support for std::string and other STL types**:
    - `HashMap` and `HashSet` now natively support `std::string` as a key type, automatically leveraging `std::hash<std::string>`.
    - Other STL types should also be supported by specializations of `std::hash<T>` for their key type.
- **New Container Features**:
    - Added `reverse()` method to `Container` interface, returning a reverse iterator for convenient backwards traversal.
    - Standardized `Map` and `Set` APIs to use references where possible, improving performance and reducing unnecessary copying.
- **New Hash Function**:
    - Added an AVX2-accelerated `excessiveFastHash` for fast hashing without dependencies on compatible hardware.
    - This hash is deliberately not randomized per run to keep hashes consistent across runs.

## Improvements

- **`sp<T>` Ownership Logic**:
    - Refined `UNIQUE` pointer behavior: copying a `UNIQUE` pointer now correctly creates a `COPY_ON_WRITE` copy while the original remains `UNIQUE`.
    - Updated documentation and comments in `pointer.h` to reflect these ownership transitions.
    - This was the original intended behavior but there was some documentation confusion.
- **Code Quality & Maintenance**:
    - Removed most usages of the `auto` keyword in favor of explicit types to improve code clarity.
    - Fixed a bug in `btree_node_t` where a default constructor was causing compiler ambiguity.
    - Standardized indentation and code style across several test suites.

## Testing

- **New Hash Quality Suite**:
    - Added comprehensive tests for `excessiveFastHash` validating:
        - **Speed**: Processing 5MiB of data in under 12ms (achieving 0.7 GiB/s locally, 1.39 in CI).
        - **Collision Resistance**: Zero collisions found across all 140,608 combinations of 3-letter strings.
        - **Avalanche Effect**: Average bit difference exceeding 31 bits for both random mutations and single-bit changes.
- **Smart Pointer Conversion Tests**:
    - Added `SpConversionTest` to verify polymorphic assignments and move operations for `sp<T>`.

# Release Notes - v0.2.0

## Breaking Changes

- **HashMap/HashSet Internal Layout**: Refactored to use a separate array model (keys, values, and bitmask) for improved
  cache locality and memory efficiency.  This migrates from using pointers for HashMap values to directly storing the
  values in a block of memory - this breaks usage of virtual classes as the value type.  It is recommended to use
  `sp<T>`, `unique_ptr<T>`, or even a basic pointer if you need to use virtual classes as the value type.  This change
  was made to improve performance, as most usages involve POD types or basic classes.

## New Features

- **Standardized Container Interface**:
    - Migrated most containers to the new `Container` base class and added iterators.
      - Standardizes containers such that different containers can be used interchangeably when
        passed to a method
- **New Integer Types**:
    - Added `uint384_t` and `uint512_t` for larger integer types.
- **Enhanced Iterators**:
    - Added iterators for `LinkedList`, `ArrayList`, `HashMap`, `HashSet`, and `SortedSet`.
    - Support for `const_iterator` and `reverse_iterator`.

## Improvements

- **Documentation**:
    - Added comprehensive Doxygen comments to many files across the codebase.
- **Code Quality**:
    - Added license comments to source files and a new `LICENSE` file.
- **Bug Fixes**:
    - Fixed severe issues due to missing includes in some headers.

## Testing

- **Enhanced Test Coverage**:
    - Expanded testing for container iterators.
    - Added missing `LongKey` tests, which were present locally but not checked in to git.
