/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-03-05
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


#include <ds/HashMap.h>
#include <atomic>
#include "alloc/pointer.h"
#include "alloc/MemoryManager.h"

/*
typedef struct {
	std::atomic<uint64_t> refs;
} allocation_data_t;

typedef struct {
	Allocator* allocator;
	HashMap<uint32_t, allocation_data_t*> allocations;
	std::mutex mutex;
	uint32_t count;
} allocator_data_t;


static Allocator& spAllocator() {
	static ModuleLevelAllocator* allocator = nullptr;

	if (allocator == nullptr)
		allocator = &memoryManager.createModule("__sp_tracker");

	return *allocator;
}


static HashMap<uint16_t, allocator_data_t*>* getAllocationMap() {
	static HashMap<uint16_t, allocator_data_t*>* map = nullptr;

	if (map == nullptr) {
		Allocator& allocator = spAllocator();
		map = allocator.allocate(HashMap<uint16_t, allocator_data_t*>(128, allocator));
	}

	return map;
}

static HashMap<Allocator*, uint16_t>* getAllocatorToId() {
	static HashMap<Allocator*, uint16_t>* map = nullptr;

	if (map == nullptr) {
		Allocator& allocator = spAllocator();
		map = allocator.allocate(HashMap<Allocator*, uint16_t>(128, allocator));
	}

	return map;
}

static uint16_t allocatorCount = 0;
static std::mutex allocatorMutex;


int compressed_pointer_details::initialize(Allocator& allocator, size_t size) {
	Allocator& spMemory = spAllocator();
	HashMap<uint16_t, allocator_data_t*>* idToAllocator = getAllocationMap();
	HashMap<Allocator*, uint16_t>* allocatorToId = getAllocatorToId();
	allocator_data_t* allocatorData;

	{
		std::lock_guard _(allocatorMutex);

		allocatorIdx = allocatorToId->getOrDefault(&allocator, 0);
		if (allocatorIdx == 0) {
			// Increment is done first so I can use zero to indicate empty/unassigned/null
			allocatorIdx = ++allocatorCount;
			allocatorData = spMemory.allocateArray<allocator_data_t>(1);
			new (allocatorData) allocator_data_t{&allocator, HashMap<uint32_t, allocation_data_t*>(256), {}, 1};
			allocatorToId->put(&allocator, allocatorIdx);
			idToAllocator->put(allocatorIdx, allocatorData);
		} else {

		}
	}

	return 0;
}

void *compressed_pointer_details::decode() {
	return nullptr;
}

void compressed_pointer_details::inc() {

}

void compressed_pointer_details::dec() {

}
*/