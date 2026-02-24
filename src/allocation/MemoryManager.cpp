//
// Created by nuclaer on 10.9.25.
//

#include "allocation/MemoryManager.h"
#include <sys/sysinfo.h>


size_t checkFreeSystemMemoryBytes() {
	struct sysinfo info;
	sysinfo(&info);
	return info.freeram * info.mem_unit;
}


MemoryManager memoryManager;
Allocator& defaultAllocator = memoryManager.getDefaultModule();


ModuleLevelAllocator::ModuleLevelAllocator(MemoryManager& manager, const char* name)
	: manager(manager), name(name), heapOffsetToSize(128), ptrToSize(1000000) {

}

ModuleLevelAllocator& MemoryManager::createModule(const char* name) {
	ModuleLevelAllocator* allocator = allocators[0];
	if (numAllocators < MAX_ALLOCATORS)
		allocator = allocators[numAllocators++] = new ModuleLevelAllocator(*this, name);

	return *allocator;
}

Range<ModuleLevelAllocator**> MemoryManager::allModules() {
	return Range<ModuleLevelAllocator**>(allocators, &allocators[numAllocators]);
}

void* ModuleLevelAllocator::alloc(size_t size) {
	// 16 is for the extra use by malloc/free
	totalMemoryUsed += ((15 + size) & 0xFFFFFFFFFFFFFFF0LU) + 16;
	void* ptr = malloc(size);
	ptrToSize.put(ptr, size);
	return ptr;

	//return slab.alloc(size);
}

void ModuleLevelAllocator::free(const void* ptr) {
	totalMemoryUsed -= ptrToSize.remove(ptr);
	::free((void*)ptr);
	//Allocator::free(ptr);
}

bool ModuleLevelAllocator::freeIfPresent(const void *ptr) {
	return false;
}

void ModuleLevelAllocator::prepare(int count, int size) {

}

void ModuleLevelAllocator::cleanup() {
	slab.destroyUnallocatedChunks();
}

size_t ModuleLevelAllocator::getBytesControlled() {
	size_t total = slab.getBytesControlled();
	/*for (uint32_t i = 0; i < heapOffsetToSize.numUsed(); i++) {
		if (heapOffsetToSize)
	}*/
	return total;
}

size_t ModuleLevelAllocator::getBytesAllocated() {
	return totalMemoryUsed;
}

uint32_t ModuleLevelAllocator::countBlocks() {
	return 0;
}

uint32_t ModuleLevelAllocator::countChunks() {
	return 0;
}

bool ModuleLevelAllocator::contains(const void *ptr) {
	return false;
}

void ModuleLevelAllocator::freeAll() {

}

ModuleLevelAllocator::~ModuleLevelAllocator() {

}
