# Bigint: Fixed-Width Large Integers

LibExcessive provides fixed-width large integer types (e.g., 128, 256 bits), optimized for applications requiring exact
bit-widths and non-allocating arithmetic.

## Characteristics

- **Zero-Allocation Arithmetic**: All operations occur inline, avoiding heap allocations common in other bigint
  libraries.
- **Fixed-Width Performance**: Predictable memory layout and performance, behaving like standard C++ types (`uint64_t`).
- **Standard Operator Overloading**: Full support for arithmetic, bitwise, and comparison operators.
- **Math Extensions**: Includes support for exponentiation (`pow`), roots (`root`), and direct multiplication with
  `double`.

## Usage

### Arithmetic and Hex Strings

```cpp
#include "bigint.h"
#include <cstdio>

void bigintExample() {
    // Construct from hex strings
    uint256_t a("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    uint256_t b("0x1234567890ABCDEF1234567890ABCDEF");

    uint256_t sum = a + b;
    uint256_t prod = a * b;

    // Advanced math
    uint256_t b_pow_4 = b.pow(4);
    uint256_t a_sqrt = a.root(2);

    printf("Sum (hex): %s\n", sum.toHexString().c_str());
}
```

### Direct Multiplication with Double

Multiplication with a `double` is floored and allows for efficient scaling of large values.

```cpp
void scaleValue() {
    uint192_t val("0x6935282358963433459348abcdef1ee7");
    uint192_t scaled = val * 0.7311;
}
```

## API Reference

### Common Types

- `uint128_t` (aliased to `UnsignedFixedWidthBigInt<2>`)
- `uint192_t` (aliased to `UnsignedFixedWidthBigInt<3>`)
- `uint256_t` (aliased to `UnsignedFixedWidthBigInt<4>`)

### Operations

- **Construction**: `BigInt(uint64_t)`, `BigInt(hex_string)`, `BigInt(double)`, `BigInt(raw_bytes, isBigEndian)`.
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`.
- **Bitwise**: `&`, `|`, `^`, `~`, `<<`, `>>`.
- **Comparison**: `<`, `>`, `<=`, `>=`, `==`, `!=`.
- **Functions**: `pow(n)`, `root(n)`, `sqrt()`, `toHexString()`, `toDouble()`.
