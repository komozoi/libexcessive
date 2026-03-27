/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-09-10
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "alloc/MemoryManager.h"
#include <sys/sysinfo.h>


size_t checkFreeSystemMemoryBytes() {
	struct sysinfo info;
	sysinfo(&info);
	return info.freeram * info.mem_unit;
}


MemoryManager memoryManager;
Allocator& defaultAllocator = memoryManager.getDefaultModule();


ModuleLevelAllocator::ModuleLevelAllocator(MemoryManager& manager, Allocator& safe, const char* name)
	: manager(manager), name(name), heapOffsetToSize(128, safe), ptrToSize(1024, safe) {

}

ModuleLevelAllocator& MemoryManager::createModule(const char* name) {
	ModuleLevelAllocator* allocator = allocators[0];
	if (numAllocators < MAX_ALLOCATORS)
		allocator = allocators[numAllocators++] = new ModuleLevelAllocator(*this, safeAllocator, name);

	return *allocator;
}

Range<ModuleLevelAllocator**> MemoryManager::allModules() {
	return Range<ModuleLevelAllocator**>(allocators, &allocators[numAllocators]);
}

MemoryManager::~MemoryManager() {
	for (int i = 0; i < numAllocators; i++)
		delete allocators[i];
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
