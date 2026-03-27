/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2023-05-08
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


#include <cstdio>
#include "alloc/SlabAllocator.h"


// Allows us to use valgrind to find memory leaks
//#define DEBUG_MEMORY 1


SlabAllocator slab;


static short countSetBits(uint32_t v) {
	// Vruuka https://stackoverflow.com/questions/14555607/number-of-bits-set-in-a-number
	// Ken pry cauu, vruuka http://graphics.stanford.edu/~seander/bithacks.html
	v = v - ((v >> 1) & 0x55555555);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
	return (short)((((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24);
}


static int blockIdByPtr(const slab_chunk_t* chunk, const char* ptr) {
	if (ptr < chunk->blocks)
		return -1;
	size_t distance = ptr - chunk->blocks;
	if (distance % chunk->blockWidth)
		return -1;
	return (int)(distance / chunk->blockWidth);
}

static slab_chunk_t* createSlabChunk(size_t blockSize, int numBlocks) {
	// Allocate blocks
	void* blocks = malloc(blockSize * numBlocks);
	if (blocks == nullptr)
		return nullptr;

	// Compute the object size and allocate the chunk's header
	size_t neededSize = sizeof(slab_chunk_t) + sizeof(int) * ((numBlocks - 1) / 32 + 1);
	slab_chunk_t* newSlabChunk = (slab_chunk_t*) malloc(neededSize);
	if (newSlabChunk == nullptr) {
		free(blocks);
		return nullptr;
	}

	// Mark all blocks as free, then set up the object
	for (int i = 0; i < (numBlocks - 1) / 32; i++)
		newSlabChunk->freeBlocks[i] = 0xFFFFFFFFU;
	newSlabChunk->freeBlocks[(numBlocks - 1) / 32] = 0xFFFFFFFFU >> ((32 - numBlocks) % 32);
	newSlabChunk->numBlocks = numBlocks;
	newSlabChunk->lastFreeBlockIndex = 0;
	newSlabChunk->blockWidth = blockSize;
	newSlabChunk->blocks = (char*)blocks;

	return newSlabChunk;
}

static inline int getNextFreeBlockId(slab_chunk_t* chunk) {
	int n = (chunk->numBlocks - 1) / 32 + 1;
	for (int i = chunk->lastFreeBlockIndex; i < n; i++) {
		uint32_t mask = chunk->freeBlocks[i];
		if (mask)
			return 32*i + __builtin_ctz(mask);
	}

	return -1;
}


SlabAllocator::SlabAllocator() {

}

void* SlabAllocator::alloc(size_t size) {
	std::lock_guard<std::mutex> _(lock);

	slab_chunk_t* chunk = getNextChunkForSize(size);
	if (!chunk)
		return nullptr;
	int nextFreeBlockId = getNextFreeBlockId(chunk);
	if (nextFreeBlockId < 0)
		return nullptr;

	chunk->freeBlocks[nextFreeBlockId / 32]  &= 0xFFFFFFFEU << (nextFreeBlockId & 0x1F);
	if (chunk->freeBlocks[nextFreeBlockId / 32])
		chunk->lastFreeBlockIndex = nextFreeBlockId / 32;
	else if ((nextFreeBlockId | 0x1F) + 1 < chunk->numBlocks && chunk->freeBlocks[nextFreeBlockId / 32 + 1])
		chunk->lastFreeBlockIndex = (nextFreeBlockId / 32 + 1);
	else
		chunk->lastFreeBlockIndex = 0;
	return (void*)&(chunk->blocks[nextFreeBlockId * chunk->blockWidth]);
}

void SlabAllocator::free(const void* ptr) {
	if (ptr == nullptr)
		return;

	{
		std::lock_guard<std::mutex> _(lock);
		slab_chunk_t *chunk = getChunkOfMemory(ptr);

		if (chunk == nullptr) {
			dprintf(2, "Attempt to free non-slab pointer %p from SlabAllocator.  Abort.\n", ptr);
#ifndef DEBUG_MEMORY
			abort();
#else
			*(char *) ptr = 1;  // Indicates invalid free
			return;
#endif
		}

		internalFree(chunk, ptr);
	}

#ifdef DEBUG_MEMORY
	destroyUnallocatedChunks();
#endif
}

bool SlabAllocator::freeIfPresent(const void* ptr) {
	if (ptr == nullptr)
		return false;

	std::lock_guard<std::mutex> _(lock);
	slab_chunk_t* chunk = getChunkOfMemory(ptr);
	if (chunk == nullptr)
		return false;

	internalFree(chunk, ptr);
	return true;
}

void SlabAllocator::freeAll() {
	std::lock_guard<std::mutex> _(lock);

	chunks.resetCursor();
	while (chunks.isCursorValid()) {
		slab_chunk_t* chunk = chunks.getCursor();

		::free(chunk->blocks);
		::free(chunk);
		chunks.remove();
	}
}

bool SlabAllocator::contains(const void* ptr) {
	std::lock_guard<std::mutex> _(lock);

	return getChunkOfMemory(ptr) != nullptr;
}

void SlabAllocator::internalFree(slab_chunk_t* chunk, const void* ptr) {
	int blockId = blockIdByPtr(chunk, (char*)ptr);
	if (blockId < 0) {
		dprintf(2, "Attempt to free bad pointer %p from SlabAllocator.  Abort.\n", ptr);
#ifndef DEBUG_MEMORY
		abort();
#else
		*(char*)ptr = 1;  // Indicates invalid free
		return;
#endif
	}

	if ((chunk->freeBlocks[blockId / 32] >> (blockId % 32)) & 1) {
		dprintf(2, "Attempt to free already free block %p from SlabAllocator.  Abort.\n", ptr);
#ifndef DEBUG_MEMORY
		abort();
#else
		*(char*)ptr = 1;  // Indicates invalid free
		return;
#endif
	}

	chunk->freeBlocks[blockId / 32] |= 1 << (blockId & 0x1F);
	chunk->lastFreeBlockIndex = blockId / 32;
}

slab_chunk_t* SlabAllocator::getChunkOfMemory(const void* ptr) {
	chunks.resetCursor();
	while (chunks.isCursorValid()) {
		slab_chunk_t* chunk = chunks.next();
		if (chunk->blocks <= ptr && &chunk->blocks[chunk->blockWidth * chunk->numBlocks] > ptr)
			return chunk;
	}

	return nullptr;
}

slab_chunk_t* SlabAllocator::getNextChunkForSize(size_t size) {
	int cntOfSize = 0;
	size_t blockSize = (size + 15) & 0xFFFFFFFFFFFFFFF0LU;

#ifndef DEBUG_MEMORY
	chunks.resetCursor();
	while (chunks.isCursorValid()) {
		slab_chunk_t* chunk = chunks.next();
		if (chunk->blockWidth == blockSize) {
			for (int i = chunk->lastFreeBlockIndex; i < chunk->numBlocks / 32; i++)
				if (chunk->freeBlocks[i])
					return chunk;
			cntOfSize += chunk->numBlocks;
		}
	}
#endif

#ifdef DEBUG_MEMORY
	int numNewBlocks = 1;
#else
	int numNewBlocks = cntOfSize * 3;  // Quadruples the amount allocated to this size
	if (numNewBlocks < 32)
		numNewBlocks = size <= 32 ? 128 : 32;
#endif

	slab_chunk_t* newSlabChunk = createSlabChunk(blockSize, numNewBlocks);
	if (!newSlabChunk) {
		// Running out of memory, clean up and try again with less new blocks
		destroyUnallocatedChunks();
		newSlabChunk = createSlabChunk(blockSize, cntOfSize);
		if (!newSlabChunk)
			// Out of memory
			return nullptr;
	}

	chunks.insertAtBeginning(newSlabChunk);

	return newSlabChunk;
}

void SlabAllocator::prepare(int count, int size) {
	std::lock_guard _(lock);

	if (count > 16) {
		slab_chunk_t *chunk = createSlabChunk(size, (count + 15) & 0xFFFFF0);
		if (chunk)
			chunks.insertAtBeginning(chunk);
	}
}

void SlabAllocator::destroyUnallocatedChunks() {
	std::lock_guard<std::mutex> _(lock);

	chunks.resetCursor();
	while (chunks.isCursorValid()) {
		slab_chunk_t* chunk = chunks.getCursor();

		bool isFree = true;
		for (int i = 0; (i < (chunk->numBlocks - 1) / 32) && isFree; i++)
			if (chunk->freeBlocks[i] != 0xFFFFFFFFU)
				isFree = false;
		if (isFree && chunk->freeBlocks[(chunk->numBlocks - 1) / 32] == 0xFFFFFFFFU >> ((32 - chunk->numBlocks) % 32)) {
			// Chunk is completely free, destroy it
			::free(chunk->blocks);
			::free(chunk);
			chunks.remove();
		} else
			chunks.next();
	}
}

size_t SlabAllocator::getBytesControlled() {
	size_t total = 0;
	chunks.resetCursor();
	while (chunks.isCursorValid()) {
		slab_chunk_t* chunk = chunks.next();
		total += chunk->blockWidth * chunk->numBlocks;
	}
	return total;
}

size_t SlabAllocator::getBytesAllocated() {
	std::lock_guard<std::mutex> _(lock);

	size_t total = 0;
	chunks.resetCursor();
	while (chunks.isCursorValid()) {
		slab_chunk_t* chunk = chunks.next();
		int freeBlocks = 0;
		for (int i = 0; i < chunk->numBlocks / 32; i++)
			freeBlocks += countSetBits(chunk->freeBlocks[i]);
		total += chunk->blockWidth * (chunk->numBlocks - freeBlocks);
	}
	return total;
}

uint32_t SlabAllocator::countBlocks() {
	std::lock_guard<std::mutex> _(lock);

	uint32_t total = 0;
	chunks.resetCursor();
	while (chunks.isCursorValid())
		total += chunks.next()->numBlocks;
	return total;
}

uint32_t SlabAllocator::countChunks() {
	return chunks.size();
}

SlabAllocator::~SlabAllocator() {
	size_t bytesAllocated = getBytesAllocated();
	if (bytesAllocated > 0) {
		fflush(stdout);
		dprintf(2, "WARNING: Destroying SlabAllocator with %lu bytes still allocated!\n", bytesAllocated);
		dprintf(2, "Either the memory is still in use, or this is a memory leak!\n");
		dprintf(2, "Blocks still allocated:\n");
	}

	std::lock_guard<std::mutex> _(lock);

	int chunkId = 0;
	chunks.resetCursor();
	while (chunks.isCursorValid()) {
		slab_chunk_t* chunk = chunks.remove();
		if (bytesAllocated > 0) {
			int freeBlocks = 0;
			for (int i = 0; i < chunk->numBlocks / 32; i++)
				freeBlocks += countSetBits(chunk->freeBlocks[i]);
			int allocatedBlocks = chunk->numBlocks - freeBlocks;
			if (allocatedBlocks > 0)
				dprintf(2, "  Chunk %i: %i blocks of size %i\n", chunkId, allocatedBlocks, chunk->blockWidth);
		}
		::free(chunk->blocks);
#ifndef DEBUG_MEMORY
		// By not freeing these chunks in debug mode, valgrind can track and find where the leak occurred.
		::free(chunk);
#endif
		chunkId++;
	}
}
