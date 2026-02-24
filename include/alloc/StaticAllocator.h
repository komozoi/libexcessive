//
// Created by nuclaer on 2/10/23.
//

#ifndef EXCESSIVE_MEMORYPOOL_H
#define EXCESSIVE_MEMORYPOOL_H

#include <stdlib.h>
#include <stdio.h>
#include <new>
#include "alloc/Allocator.h"


/**
 * This class helps reduce the use of malloc and free, and allocates bigger chunks at a time, to help
 * prevent heap fragmentation on embedded systems.  The tradeoff is that you can't free anything
 * allocated in a StaticAllocator without freeing all of it.  Use this to manage many dynamically allocated
 * objects that are all deleted at the same time.  It is highly recommended to compute the amount of
 * memory you will need ahead of time, or at least make a guess, to improve performance.
 *
 * This can also improve cache performance for CPUs with a cache, if used correctly.
 */
class StaticAllocator : public Allocator {
public:

	inline explicit StaticAllocator(size_t size) {
		if (size > 0) {
			memory_capacity = size;
			memory = (char*) malloc(memory_capacity);
		}
	}

	inline StaticAllocator(char* memory, size_t capacity, size_t used) :
		memory_capacity(capacity), memory_used(used), memory(memory)
	{}

    void* alloc(size_t size) override {
		if (next)
			return next->alloc(size);
		if (memory == nullptr) {
			memory_capacity = 128 * size;
			memory = (char*)malloc(memory_capacity);
		} else if (size + memory_used + sizeof(StaticAllocator) > memory_capacity) {
			// Allocate a new MemoryPool
			// TODO: Avoid using this pool's memory, instead put the memory pool object in the new memory region
			next = (StaticAllocator*)&(memory[memory_used]);
			memory_used = memory_capacity;
			new (next) StaticAllocator(memory_capacity * 2);
			return next->alloc(size);
		}
		char* out = &(memory[memory_used]);
		memory_used += size;
		return out;
	}

	void prepare(size_t newSize) {
		if (memory == nullptr) {
			memory_capacity = newSize;
			memory = (char*) malloc(memory_capacity);
		}
	}

	/**
	 * Trims the memory to the size allocated over the lifetime of the object,
	 * then transfers ownership of the internal memory to the caller.
	 * @tparam T type the returned pointer should point to
	 * @return pointer to the beginning of the allocated memory
	 */
	template<class T>
	T* exportAndTrim() {
		realloc(memory, memory_used);

		if (next)
			printf("Memory leak detected in MemoryPool, by use of MemoryPool::exportAndTrim()\n");

		T* tmp = (T*)memory;
		memory = nullptr;
		return tmp;
	}

	~StaticAllocator() override {
		if (memory) {
			if (next)
				next->~StaticAllocator();
			::free(memory);
		}
	}

	void reset() {
		memory_used = 0;
		if (next)
			next->~StaticAllocator();
		next = nullptr;
	}

	bool contains(void* ptr) {
		if (ptr >= memory && ptr < &memory[memory_used])
			return true;
		else if (next)
			return next->contains(ptr);
		return false;
	}

protected:
	size_t memory_capacity = 0;
	size_t memory_used = 0;
	char* memory = nullptr;
	StaticAllocator* next{};
};


#endif //EXCESSIVE_MEMORYPOOL_H
