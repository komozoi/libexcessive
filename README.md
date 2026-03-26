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

### File Access

```cpp
#include "fs/FdHandle.h"


int main() {
    const char* temp_filename = "mmap_test.tmp";

    // All file handles are smart pointers!
    FdHandle write_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
    MmapHandle write_mmap = write_handle.getMmapHandle(0, sizeof(uint32_t));

    // Writing structs with mmap is easy and safe
    my_struct_t value{1, true, {}};
    write_mmap.write(value);

    // File automatically closes when all references go out of scope,
    // but can be closed manually with:
    // write_handle.close();
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
