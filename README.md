# LibExcessive

- High-performance, parallel file I/O with RAII semantics
- High-performance on-disk data structures
- A bunch of utility functions and classes

## Try it in 5 minutes

Create a CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.14)
project(BTreeDemo)

# Adds excessive to the project
include(FetchContent)

FetchContent_Declare(
        excessive
        GIT_REPOSITORY https://gitea.com/komozoi/excessive.git
        GIT_TAG v0.1.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS ON
        SYSTEM
)

FetchContent_MakeAvailable(excessive)

# Create the demo executable and link excessive into it
add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE excessive)
```

And create a file `demo.cpp`:

```c++
#include <fs/FdHandle.h>
#include <fs/BTree.h>
#include <fcntl.h>
#include <random>
#include <cstdio>


struct btree_entry_s {
    int key;
    int value;

    static int compare(const btree_entry_s &a, const btree_entry_s &b) {
        return a.key - b.key;
    }
};

int main() {
    FdHandle file = FdHandle::open("btree.bin", O_RDWR | O_CREAT, 0644);
	if (!file) {
		printf("Failed to open file!\n");
		return 1;
	}

    // It is easy to check if the file already existed or was just created
    printf("File is %s.\n", file.isNew() ? "newly created" : "existing");

    BTree<btree_entry_s> tree(file, 0, btree_entry_s::compare);

    // Add 100 random elements to the tree
    std::random_device rd;
    for (int i = 0; i < 100; ++i) {
        btree_entry_s new_entry{(int) rd() % 5000, (int) rd()};
        tree.insert(new_entry);
    }

    // See what the next highest values are for given inputs
    // Each run this would change as the BTree grows
    for (int i = 0; i < 5000; i += 500) {
        btree_entry_s result{i, 0};
        if (tree.findNext(result)) {
            printf("Next highest entry from %i is (%i, %i)\n", i, result.key, result.value);
        } else {
            printf("No next highest entry found for %i\n", i);
        }
    }

    // All data is already written to the file (although not necessarily flushed)
    // For this reason, no cleanup is needed here for the BTree.

    // File automatically closes when all references go out of scope,
    // but can be closed manually with:
    // write_handle.close();

	return 0;
}
```

If you are using an IDE, this may be enough to import the project and run it - very convenient!

If not, run these commands to compile and run:

```bash
# Setup
mkdir build && cd build && cmake ..

# Compile
make

# Run the demo to create the data file
./demo

# Run the demo again to add more to the data file and see the effects
./demo
```

And that's it - efficient and persistent data storage in less than 60 lines of code.  No extra installation steps
or complex APIs.  It just works.

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
    printf("b**4  = %s\n", pow.toHexString().c_str());

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

### Smart Pointer with Copy on Write Behavior

```c++
#include <alloc/pointer.h>

struct Data {
    int value;
};

int main() {
    // UNIQUE (default-style ownership)
    sp<Data> x(SpPointerType::UNIQUE, Data{10});

    // Copying a UNIQUE pointer does NOT immediately copy the data
    sp<Data> y = x;
    // x stays UNIQUE
    // y becomes COPY_ON_WRITE
    // both point to the same underlying object (for now)

    // First write triggers a deep copy
    y.mut().value = 20;
    // now:
    // x->value == 10
    // y->value == 20
    // they no longer share memory


    // You can keep copying before mutation is needed
    sp<Data> z = x;
    // still sharing with x

    z.mut().value = 30;
    // z detaches and becomes independent
    // x is still unchanged


    // SHARED mode = always shared, no copy-on-write
    sp<Data> sharedA(SpPointerType::SHARED, Data{100});
    sp<Data> sharedB = sharedA;

    sharedB.mut().value = 200;
    // both see the change:
    // sharedA->value == 200
    // sharedB->value == 200


    // Move = transfer ownership, no copies
    sp<Data> moved = std::move(sharedA);
    // sharedA is now null
    // moved owns the data


    // Scoped lifetime (RAII)
    {
        sp<Data> temp(SpPointerType::UNIQUE, Data{5});
        sp<Data> alias = temp;
        // alias is COPY_ON_WRITE

        // object is destroyed exactly once when both go out of scope
    }

    return 0;
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
