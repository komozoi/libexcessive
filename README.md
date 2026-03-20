# LibExcessive

A C++ utility library that brings Java-like container interfaces
to C++ while avoiding verbosity and staying fast, with emphasis on
making safety and data organization easy.

## Overview

LibExcessive is intended for large, data-heavy backend
applications such as servers and data processing tools where speed and reliability matter. The
design goals and features are:

* Provide familiar, Java-like APIs and richer helper types, especially containers
  * With less verbosity than Java APIs, thankfully.
* Provide rich, threadsafe, and extremely efficient utilities for interacting with files
  * Threadsafe file handles and transactions
  * Mmap handles
  * Open file reference counting
  * Utilities for keeping data on-disk
    * BTree
    * Files with dynamically allocated regions
* (planned) Utilities for building on-disk indexes and databases
* Favor explicit memory and performance control. Many components are designed to be friendly to
  custom allocators and memory pools.
  * Tracking memory separately for different components of an application, which helps
    to find memory hogs
  * Safer allocation and memory management with less heap fragmentation
* Small, focused algorithms and helpers for string handling, byte buffers, serialization, and more
* Support modern C++ compilers from C++17 and up.

Much of the code was originally written to run on the Teensy 4.1, which is extremely memory
constrained compared to our desktop computers, having a mere 1MiB of RAM.  For this reason,
there is a lot of code for tightly controlling memory usage and performance.

Everything is tested on Debian Linux currently, although
any Unix flavor should work.  I do not plan on supporting Windows.

## Building

I develop this with CLion, which imports the CMake project for me, but if you prefer to build on
the terminal, it's not hard:

```bash
git clone https://gitlab.com/Nuclear_Man_D/libexcessive.git
cd libexcessive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage Examples

Tests are included for almost everything, which you can use as a larger reference if needed.

### File Access with Mmap

```cpp
#include "fs/FdHandle.h"


int main() {
    const char* temp_filename = "mmap_test.tmp";

    // All file handles are smart pointers!
    FdHandle write_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
    MmapHandle write_mmap = write_handle.getMmapHandle(0, sizeof(my_struct_t));

    // Writing structs with mmap is easy and safe
    my_struct_t value{1, true, {}};
    write_mmap.write(value);

    // File automatically closes when all references go out of scope,
    // but can be closed manually with:
    // write_handle.close();
}
```

### Bigint

Unlike in other libraries, the bigint implementation uses a fixed width.  The bigint code was originally designed for use
in an EVM implementation, where most values are 256-bit.  This is designed to work like a typical register or even
normal fixed-width datatype, just bigger, and acts like you would expect with truncation and such.

Examples:

```c++
#include "bigint.h"

int main() {
    uint256_t a = uint256_t("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
    uint256_t b("0x1234567890ABCDEF1234567890ABCDEF");

    uint256_t sum  = a + b;
    uint256_t prod = a * b;
    uint256_t pow  = b.pow(4);

    printf("a     = %s\n", a.toHexString().c_str());
    printf("b     = %s\n", b.toHexString().c_str());
    printf("a + b = %s\n", sum.toHexString().c_str());
    printf("a * b = %s\n", prod.toHexString().c_str());
    printf("b⁴    = %s\n", pow.toHexString().c_str());

    // Output:
    // a     = ffffffffffffffffffffffffffffffff
    // b     = 1234567890abcdef1234567890abcdef
    // a + b = 0000000000000000000000001234567890abcdef1234567890abcdefffffffff
    // a * b = 1234567890abcdef1234567890abcdeeffffffffffffffedcba98765432111
    // b**4  = 33ee0e405772f4bd1fa6d7a4e8c14117ea371272c23e2b10

	// Default sizes include uint128_t, uint192_t, and uint256_t.  Custom sizes are also possible.
	uint192_t threeWordValue = "0x6935282358963433459348abcdef1ee7";
	UnsignedFixedWidthBigInt<7> sevenWordValue = "0x8b20159b1c579b1088048f054bedebfd02de6b23919371be36d872ec46fe9cebe684edd2675ab1101262b78877b3c09966366c07df0fcccf";

	// It is also possible to multiply directly with doubles
	// The result of the multiplication is floored and almost exact
	uint192_t productWithDouble = threeWordValue * 0.7311;
	// productWithDouble would be something like 0x4cead4b7be06ccb79c0e92711d09028c

	uint256_t x3("0x783924abc37678847777fcba");
	x3 = x3.root(3);  // 0xc6fc718e
}
```

### Simple Containers

Simple examples showing the style of usage that matches the library design:

```cpp
#include "ds/ArrayList.h"


int main() {
    ArrayList<int> list;
    list.add(1);
    list.add(2);
    list.addCopies(5, 3); // add three copies of 5

    for (int element: list)
        printf("value %d\n", element);

    return 0;
}
```

```cpp
#include "ds/ArrayList.h"


int main() {
    // Initialize list as {2, 3, 4}
    ArrayList<int> list{2,3,4};

    // Add 1 to the beginning
	list.addFirst(1);

    // 1, 2, 3, 4
    for (int element: list)
        printf("value %d\n", element);
}
```
