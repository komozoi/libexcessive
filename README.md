# LibExcessive

A compact, high-utility C++ support library that brings Java-like container and utility interfaces
to C++ while staying fast, deterministic, and linker-friendly.

## Overview

LibExcessive is a single, linkable C++ convenience library intended for large, data-heavy backend
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

I do not plan on supporting Windows.  Everything is tested on Debian Linux currently, although
any Unix flavor should work.

Main repository is on Gitea at https://gitea.com/komozoi/excessive, but is mirrored to GitHub
at https://github.com/komozoi/excessive.

## Building

I develop this with CLion, which imports the CMake project for me, but if you prefer to build on
the terminal, it's not hard:

```bash
git clone https://gitea.com/komozoi/excessive.git
cd libexcessive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage Examples

Small example showing the style of usage that matches the library design:

```cpp
#include "excessive/ArrayList.h"   // example header path

int main() {
    ArrayList<int> list;
    list.add(1);
    list.add(2);
    list.addCopies(5, 3); // add three copies of 5
    for (int i = 0; i < list.size(); ++i)
        printf("value %d\n", list.get(i));
    return 0;
}
```
