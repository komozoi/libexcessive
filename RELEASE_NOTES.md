
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
