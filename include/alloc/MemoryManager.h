//
// Created by nuclaer on 9/9/25.
//

#ifndef EXCESSIVE_MEMORYMANAGER_H
#define EXCESSIVE_MEMORYMANAGER_H

#include "stdint.h"
#include "alloc/Allocator.h"
#include "SlabAllocator.h"
#include "ds/HashMap.h"
#include "Range.h"

size_t checkFreeSystemMemoryBytes();


class MemoryManager;


class ModuleLevelAllocator : public Allocator {
public:
	inline ModuleLevelAllocator(MemoryManager& manager, Allocator& safe, const char* name);

	void* alloc(size_t size) override;

	/**
	 * Works like normal free(): accepts nullptr, but will abort if an invalid pointer is provided
	 * @param ptr the memory to free
	 */
	void free(const void* ptr) override;

	/**
	 * Will check if the memory is allocated in this allocator.  If it is, then it works like normal free(), but
	 * returns true.  If the pointer is not in the allocator, then it returns false and has no effect.
	 *
	 * To be clear, if an attempt is made to free memory between blocks, or an attempt to free a free block, then
	 * abort() will be called.  Obviously the function cannot return in such a case.
	 *
	 * @param ptr the memory to free
	 * @return true if the memory is in the allocator and represents a valid block, false if not in slab allocator.
	 */
	bool freeIfPresent(const void* ptr);

	/**
	 * Create a new block with at least count elements of the given size in the SlabAllocator this object controls
	 * @param count Number of blocks
	 * @param size Size of each block
	 */
	void prepare(int count, int size);

	template <class T>
	void prepare(int count) {
		prepare(count, sizeof(T));
	}

	/**
	 * Looks for ways to cleanup memory overhead and thus reduce memory consumption
	 */
	void cleanup();

	/**
	 * @return The number of bytes this object has reserved to be allocated.  This includes bytes already allocated.
	 */
	size_t getBytesControlled();

	/**
	 * @return The number of bytes allocated by alloc() and functions that use alloc() (includes clone, allocate, etc)
	 */
	size_t getBytesAllocated();

	uint32_t countBlocks();
	uint32_t countChunks();

	/**
	 * Check if a pointer is controlled by this allocator
	 * @param ptr the pointer to check
	 * @return true if this allocator allocated it, false otherwise
	 */
	bool contains(const void* ptr);

	/**
	 * Free all memory allocated using this object
	 */
	void freeAll();

	~ModuleLevelAllocator();

	MemoryManager& manager;
	const char* name;

private:
	SlabAllocator smaller;

	size_t amountAllocatedOnHeap = 0;
	HashMap<int, uint32_t> heapOffsetToSize;

	HashMap<const void*, uint32_t> ptrToSize;
	size_t totalMemoryUsed = 0;
};


/**
 * This will be a better way of managing the memory of expressions and similar objects, for which there are many that
 * are frequently created, copied, and destroyed.
 */
class MemoryManager {
public:

	inline MemoryManager() {
		createModule("default");
	}

	inline ModuleLevelAllocator& getDefaultModule() { return *allocators[0]; }
	ModuleLevelAllocator& createModule(const char* name);

	Range<ModuleLevelAllocator**> allModules();

private:
	DefaultAllocator safeAllocator;
	const static int MAX_ALLOCATORS = 32;
	ModuleLevelAllocator* allocators[MAX_ALLOCATORS];
	int numAllocators = 0;
};


extern MemoryManager memoryManager;


#endif //EXCESSIVE_MEMORYMANAGER_H
