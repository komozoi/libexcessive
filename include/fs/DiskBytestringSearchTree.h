/*
 * Copyright 2023-2026 komozoi
 * Original Creation Date: 2025-10-12
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
 *
 */

#ifndef RELIABLEKEYVALUESTORE_DISKBYTESTRINGSEARCHTREE_H
#define RELIABLEKEYVALUESTORE_DISKBYTESTRINGSEARCHTREE_H

#include "FdHandle.h"
#include "ds/Bytestring.h"
#include "FreeSpaceFile.h"


/**
 * Each byte has 256 combinations, so if we have a tree
 * where each node is a byte, then the max number of
 * children is 256.  If the node is simply an array of
 * child references, we can predict the location of the
 * next byte, saving a lot of IO and search.  Then the
 * maximum time complexity to find a bytestring with
 * N characters is O(N).  This is approximately logarithmic
 * relative to the total number of keys, as this is a
 * tree structure.
 */

/**
 * Variable-length node encoding
 * Order is the node struct, then a sorted array of uint64_ts referencing the children,
 * then the prefix bytes
 *
 * It is probably good to allocate an extra 32 bytes to leave room for future array expansion
 */
struct disk_bytestring_node_t {
	uint64_t parent;

	// Offset for the referenced data
	uint64_t dataOffset;

	// Used for allocation/deallocation purposes both on-disk and in-memory.
	uint32_t nodeMemory;

	// Any remaining part of the bytestring is written after the struct,
	// according to this length.
	// The bytestring must be checked to see if the key is an exact match.
	uint16_t prefixLength;

	uint16_t numChildren;

	// For convenience access
	uint64_t children[];

	size_t computeSize() const {
		return sizeof(disk_bytestring_node_t) + prefixLength + numChildren * 8;
	}

	const uint8_t* getPrefix() {
		return &((uint8_t*)&this[1])[numChildren * 8];
	}

	size_t write(FdTransaction& writer) const;
	static disk_bytestring_node_t* read(FdTransaction& reader);
	static disk_bytestring_node_t* create(uint64_t parent, uint64_t dataOffset, uint16_t prefixLength, const uint8_t* prefix, uint16_t preallocatedChildren);

	// These use binary search
	uint64_t findChildByByte(uint8_t nextByte);
	disk_bytestring_node_t* addChild(uint8_t nextByte, uint64_t childOffset);

	void migrateChild(uint64_t oldOffset, uint64_t newOffset);

	void updatePrefix(const uint8_t *newPrefix, uint32_t length);
};


class DiskBytestringSearchTree {
public:
	DiskBytestringSearchTree(FreeSpaceFile& fss, uint64_t rootOffset);

	static uint64_t initialize(FreeSpaceFile& fss);

	/**
	 * Insert a new value to the index, if no matching key is found, or if overwrite is set.
	 * If a matching key is found, then no change is made.
	 * @param key Bytestring key
	 * @param value value for the given key
	 * @param overwrite weather to overwrite the value if it is present
	 * @return true if the key is already present, false if the key was not previously present and has been inserted
	 */
	bool insert(const Bytestring& key, uint64_t value, bool overwrite = false);

	/**
	 * Finds the given value, if present.
	 * @param key key to search for
	 * @return true if the value was found, false if not
	 */
	uint64_t find(const Bytestring& key);

	/**
	 * Finds the given value, if present.
	 * If found, will remove the element from the tree.
	 * @param key key to search for
	 * @return true if the value was found and deleted, false if not
	 */
	uint64_t remove(const Bytestring& key);

	void flush();

#ifdef IS_GTEST
	int interruptionNext = -1;
#endif

private:
	FreeSpaceFile& fss;
	uint64_t rootOffset;

	enum MatchType {
		INVALID_KEY,
		EMPTY_ENTRY,
		EXACT,
		TO_SPLIT
	};

	MatchType findNearestNode(const Bytestring& key, uint64_t& offset, uint8_t& nodeIndex, uint32_t& keyIndex, disk_bytestring_node_t*& node);

	uint64_t createLeaf(FdTransaction& tx, uint64_t parent, const uint8_t* prefix, uint16_t prefixLength, uint64_t value);
};


#endif //RELIABLEKEYVALUESTORE_DISKBYTESTRINGSEARCHTREE_H
