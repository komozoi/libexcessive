# Release Notes - v0.2.0 (Changes since v0.1.0)

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
