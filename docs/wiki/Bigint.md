---
title: "Bigint: Fixed-Width Large Integers"
description: "High-performance, non-allocating fixed-width large integer arithmetic for 128, 256, 512 bits and beyond."
keywords: "bigint, uint256_t, fixed-width, cryptography, performance, c++"
category: "Data Types"
---

# Bigint: Fixed-Width Large Integers

LibExcessive provides a suite of fixed-width large integer types (e.g., 128, 256, 512 bits) optimized for
high-performance applications that require exact bit-widths without the overhead of dynamic heap allocations. These
types behave similarly to built-in C++ integer types but support much larger ranges.

## Characteristics

`UnsignedFixedWidthBigInt` is designed to be a drop-in replacement for standard integers in performance-critical code:

- **Zero-Allocation Arithmetic**: All operations occur inline using stack-allocated memory. No heap allocations are
  performed during arithmetic, bitwise, or comparison operations.
- **Fixed-Width Performance**: The memory layout is predictable and contiguous, consisting of an array of `uint64_t`
  "chunks."
- **Comprehensive Operator Overloading**: Supports all standard C++ arithmetic (`+`, `-`, `*`, `/`, `%`), bitwise (`&`,
  `|`, `^`, `~`, `<<`, `>>`), and comparison (`<`, `>`, `==`, etc.) operators.
- **Advanced Math Extensions**: Built-in support for exponentiation (`pow`), roots (`root`, `sqrt`), and signed
  operations (even though the type is technically unsigned).
- **Floating-Point Integration**: Supports direct multiplication with `double` and conversion to/from `double` with high
  precision.

## Usage Example

```cpp
#include "bigint.h"
#include <iostream>

void bigintExample() {
    // Construct from hex strings
    uint256_t a("0xFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    uint256_t b("0x1234567890ABCDEF1234567890ABCDEF");

    // Standard arithmetic
    uint256_t sum = a + b;
    uint256_t prod = a * b;

    // Advanced math
    uint256_t b_pow_4 = b.pow(4);
    uint256_t a_sqrt = a.root(2);

    // Conversion to string (produces "0x1204567890ABCDEF1234567890ABCDEE")
    std::cout << "Sum: " << std::string(sum) << std::endl;
}
```

## API Reference: UnsignedFixedWidthBigInt<N>

The template parameter `N` defines the number of 64-bit chunks (e.g., `N=4` for a 256-bit integer).

### Common Type Aliases

- `uint128_t`: `UnsignedFixedWidthBigInt<2>`
- `uint192_t`: `UnsignedFixedWidthBigInt<3>`
- `uint256_t`: `UnsignedFixedWidthBigInt<4>`
- `uint384_t`: `UnsignedFixedWidthBigInt<6>`
- `uint512_t`: `UnsignedFixedWidthBigInt<8>`

### Constructors

- `UnsignedFixedWidthBigInt()`
    - Default constructor. The value is left uninitialized for performance.
- `UnsignedFixedWidthBigInt(uint64_t value)`
    - Constructs from a standard 64-bit unsigned integer.
- `UnsignedFixedWidthBigInt(int value)`
    - Constructs from a signed integer. Handles negative values via two's complement.
- `UnsignedFixedWidthBigInt(const char* s)` / `UnsignedFixedWidthBigInt(std::string_view s)`
    - Parses a string (optionally prefixed with `0x`) as a hexadecimal number.
- `UnsignedFixedWidthBigInt(double value)`
    - Constructs from a double-precision floating-point value.
- `UnsignedFixedWidthBigInt(const uint8_t* buffer, bool isBigEndian)`
    - Copies raw bytes from a buffer.

### Arithmetic Operators

- `operator+(const UnsignedFixedWidthBigInt<N>& other) const`
- `operator-(const UnsignedFixedWidthBigInt<N>& other) const`
- `operator*(const UnsignedFixedWidthBigInt<N>& other) const`
- `operator/(const UnsignedFixedWidthBigInt<N>& other) const`
- `operator%(const UnsignedFixedWidthBigInt<N>& other) const`
    - Perform standard modular arithmetic. Division and modulo use an optimized multi-word division algorithm.

### Bitwise Operators

- `operator&(const UnsignedFixedWidthBigInt<N>& other) const`
- `operator|(const UnsignedFixedWidthBigInt<N>& other) const`
- `operator^(const UnsignedFixedWidthBigInt<N>& other) const`
- `operator~() const`
- `operator<<(int n) const` / `operator>>(int n) const`
    - Standard bitwise operations. Shift operations handle shifts larger than 64 bits across multiple chunks.

### Math Functions

- `UnsignedFixedWidthBigInt<N> pow(int power) const`
    - Computes the value raised to the given power using the binary exponentiation algorithm.
- `UnsignedFixedWidthBigInt<N> root(int root) const`
    - Computes the n-th integer root.
- `UnsignedFixedWidthBigInt<N> sar(int n) const`
    - Arithmetic right shift (preserves the sign bit).
- `void setBit(int index)` / `void resetBit(int index)` / `bool getBit(int index) const`
    - Direct manipulation of individual bits by index.
- `uint16_t countBits() const`
    - Returns the number of bits required to represent the current value (position of the highest set bit).

### Conversions and Output

- `operator std::string() const`
    - Converts the value to a hexadecimal string prefixed with `0x`. Leading zeros are omitted.
- `void toStr(char* buffer, bool forceSize = false, bool addPrefix = true) const`
    - Writes the hexadecimal representation into a provided buffer. `forceSize` ensures the output matches the full
      width of the type, in which case there may be a large number of leading zeros.  `addPrefix` adds the `0x` prefix.
- `double toDouble() const`
    - Converts the large integer to a `double`.
- `operator uint64_t() const` (and `uint32_t`, `int`)
    - Returns the lowest 64 bits (or 32 bits) of the value.

---

## How it Works: Technical Overview

`UnsignedFixedWidthBigInt` implements multi-precision arithmetic by treating a large integer as a base-$2^{64}$ number.

### Memory Representation

The internal state is stored in a union containing an array of `uint64_t` (the chunks) and a corresponding raw byte
array. Chunks are stored in little-endian order (least significant chunk at index 0), which aligns with common CPU
architectures like x86_64. This layout allows for efficient indexing and interaction with standard integer registers.

### Arithmetic Implementation

- **Addition/Subtraction**: Implemented using carry/borrow propagation across chunks. The library uses optimized
  assembly-friendly loops to minimize the cost of propagating carries.
- **Multiplication**: Uses a multi-word multiplication algorithm (similar to the grade-school method but in
  base-$2^{64}$). For small `N`, this is highly efficient.
- **Division**: Employs a robust multi-word division algorithm to handle large divisors.
- **Performance Optimization**: Many operations are implemented as "fast" internal functions (e.g.,
  `_internalBigintFastAdd`) that operate directly on pointers, which the compiler can often inline or vectorize.

### Floating-Point Interop

The `toDouble` and `fromDouble` conversions work by extracting the most significant bits of the large integer and
adjusting the exponent of the IEEE 754 double-precision format. Multiplication with a `double` (`operator*`) is
performed by scaling the large integer in its base-$2^{64}$ form, effectively performing a fixed-point multiplication
where the `double` provides the scaling factor.  This makes double multiplication extremely accurate.
