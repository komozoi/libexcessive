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

#include "fs/DiskBytestringSearchTree.h"

#include "string.h"
#include "unistd.h"


#ifdef IS_GTEST
#define CHECK_INTERRUPTION(X, Y) \
	if (interruptionNext == X) { \
		fss.getFile().close(); \
        interruptionNext = -1; \
		return Y; \
	}

#else
#define CHECK_INTERRUPTION(X, Y)
#endif


size_t disk_bytestring_node_t::write(FdTransaction& writer) const {
	return writer.write(this, computeSize());
}

disk_bytestring_node_t* disk_bytestring_node_t::read(FdTransaction& reader) {
	disk_bytestring_node_t tmp;
	reader.read(tmp);

	// Allocating extra space dramatically decreases the likelyhood of using realloc() later.
	if (tmp.nodeMemory < tmp.computeSize())
		tmp.nodeMemory = tmp.computeSize() + 32;

	disk_bytestring_node_t* out = (disk_bytestring_node_t*) malloc(tmp.nodeMemory);
	*out = tmp;

	reader.read(&out[1], tmp.numChildren * 8 + tmp.prefixLength);

	return out;
}

disk_bytestring_node_t* disk_bytestring_node_t::create(uint64_t parent, uint64_t dataOffset, uint16_t prefixLength, const uint8_t* prefix, uint16_t preallocatedChildren) {
	uint32_t nodeMemory = sizeof(disk_bytestring_node_t) + prefixLength + preallocatedChildren * 8;
	disk_bytestring_node_t* out = (disk_bytestring_node_t*) malloc(nodeMemory);
	*out = {parent, dataOffset, nodeMemory, prefixLength, 0};
	if (prefixLength)
		memcpy(&out[1], prefix, prefixLength);
	return out;
}

uint64_t disk_bytestring_node_t::findChildByByte(uint8_t nextByte) {
	int left = 0;
	int right = numChildren - 1;

	while (left <= right) {
		int mid = (left + right) / 2;
		uint8_t byte = children[mid] & 0xFF;

		if (byte == nextByte) {
			// Return the offset (upper 56 bits)
			return children[mid] >> 8;
		} else if (byte < nextByte) {
			left = mid + 1;
		} else {
			right = mid - 1;
		}
	}

	return 0;
}

disk_bytestring_node_t* disk_bytestring_node_t::addChild(uint8_t nextByte, uint64_t childOffset) {
	uint32_t memoryNeeded = computeSize() + 8;

	disk_bytestring_node_t* tmp = this;
	if (memoryNeeded > nodeMemory) {
		tmp = (disk_bytestring_node_t*)realloc(this, memoryNeeded + 16);
		if (tmp == nullptr)
			return nullptr;
		tmp->nodeMemory = memoryNeeded + 16;
	}

	// Binary search to find insertion point
	int left = 0, right = tmp->numChildren - 1;
	int insertPos = tmp->numChildren;

	while (left <= right) {
		int mid = (left + right) / 2;
		uint8_t b = tmp->children[mid] & 0xFF;
		if (b == nextByte) {
			tmp->children[mid] = (childOffset << 8) | nextByte;
			return tmp;
		} else if (b < nextByte) {
			left = mid + 1;
		} else {
			insertPos = mid;
			right = mid - 1;
		}
	}

	// Move prefix (after children) forward 8 bytes, before it gets overwritten
	uint8_t* oldPrefix = (uint8_t*)tmp->children + tmp->numChildren * sizeof(uint64_t);
	uint8_t* newPrefix = (uint8_t*)tmp->children + (tmp->numChildren + 1) * sizeof(uint64_t);

	memmove(newPrefix, oldPrefix, tmp->prefixLength);

	// Shift children to make room
	memmove(&tmp->children[insertPos + 1], &tmp->children[insertPos], (tmp->numChildren - insertPos) * sizeof(uint64_t));

	// Insert new child
	tmp->children[insertPos] = (childOffset << 8) | nextByte;

	tmp->numChildren += 1;

	return tmp;
}

void disk_bytestring_node_t::migrateChild(uint64_t oldOffset, uint64_t newOffset) {
	for (int i = 0; i < numChildren; i++)
		if (children[i] >> 8 == oldOffset) {
			children[i] &= 0xFF;
			children[i] |= newOffset << 8;
			break;
		}
}

void disk_bytestring_node_t::updatePrefix(const uint8_t* newPrefix, uint32_t length) {
	prefixLength = length;
	if (length > 0)
		memmove(&((uint8_t*)&this[1])[numChildren * 8], newPrefix, length);
}


DiskBytestringSearchTree::DiskBytestringSearchTree(FreeSpaceFile& _fss, uint64_t _rootOffset) : fss(_fss), rootOffset(_rootOffset) {
}

uint64_t DiskBytestringSearchTree::initialize(FreeSpaceFile& fss) {
	// Fill first 4096 bytes after the free space header with zeros
	disk_bytestring_node_t* root = disk_bytestring_node_t::create(0, 0, 0, nullptr, 256);

	FdTransaction tx(fss.getFile());
	if (tx.seek(0, SEEK_END) < fss.getHeaderEnd()) {
		ftruncate(fss.getFile().getFd(), fss.getHeaderEnd());
	}

	uint64_t offset = fss.getFreeRegion(root->nodeMemory);
	tx.seek(offset);
	root->write(tx);

	free(root);
	return offset;
}

bool DiskBytestringSearchTree::insert(const Bytestring& key, uint64_t value, bool overwrite) {
	uint64_t nodeOffset;
	uint8_t nodeIndex;
	uint32_t keyIndex;
	disk_bytestring_node_t* node;

	MatchType matchType = findNearestNode(key, nodeOffset, nodeIndex, keyIndex, node);

	if (matchType == INVALID_KEY) {
		free(node);
		return false;
	}

	if (matchType == EXACT) {
		uint64_t previousOffset = node->dataOffset;
		if (previousOffset != 0 && !overwrite) return true;

		node->dataOffset = value;

		// Update node with new data offset
		// We don't need to update the array or prefix,
		// so just write the struct itself
		FdTransaction tx(fss.getFile());
		tx.seek(nodeOffset);
		tx.write(*node);

		CHECK_INTERRUPTION(0, false);

		free(node);
		return previousOffset != 0;

	} else if (matchType == MatchType::EMPTY_ENTRY) {
		FdTransaction tx(fss.getFile());

		uint64_t newNodeOffset = createLeaf(tx, nodeOffset, &key[keyIndex], key.size() - keyIndex, value);

		CHECK_INTERRUPTION(1, false);

		// Write new child offset
		uint32_t nodeMemoryAllocated = node->nodeMemory;
		node = node->addChild(nodeIndex, newNodeOffset);

		if (nodeMemoryAllocated == node->nodeMemory) {
			tx.seek(nodeOffset);
			node->write(tx);
		} else {
			// Original memory block was too small - reallocate
			uint64_t newOffset = fss.getFreeRegion(node->nodeMemory);
			tx.seek(newOffset);
			node->write(tx);

			// Update the parent
			tx.seek(node->parent);
			disk_bytestring_node_t* parent = disk_bytestring_node_t::read(tx);
			parent->migrateChild(nodeOffset, newOffset);
			tx.seek(node->parent);
			parent->write(tx);
			free(parent);

			// Free the old block
			fss.markFreeRegion(nodeOffset, nodeMemoryAllocated);
		}

		CHECK_INTERRUPTION(2, false);

		free(node);
		return false;

	} else /*if (matchType == MatchType::TO_SPLIT)*/ /* always true since it is the last option */ {
		FdTransaction tx(fss.getFile());

		const uint8_t* oldPrefix = node->getPrefix();

		// Find first mismatched byte
		uint32_t mismatchAt = 0;
		while (mismatchAt < node->prefixLength && keyIndex + mismatchAt < key.size() && key[keyIndex + mismatchAt] == oldPrefix[mismatchAt])
			mismatchAt++;

		// Build new parent node with common prefix
		// This will be written to nodeOffset - the original node location.
		disk_bytestring_node_t* newParent = disk_bytestring_node_t::create(node->parent, 0, mismatchAt, oldPrefix, 4);
		uint8_t childIndex = oldPrefix[mismatchAt];

		// Update original node prefix (truncate by mismatchAt)
		// Skip first byte as it is matched by our child array instead
		// Also update the parent.  Since we are writing the parent node to this
		// node's former position, we can set the parent offet to this old offset.
		node->updatePrefix(&oldPrefix[mismatchAt + 1], node->prefixLength - mismatchAt - 1);
		node->parent = nodeOffset;

		// Write the modified original node to a new sector
		// If this flushes then the program is interrupted, then a sector will be wasted,
		// but the tree will be intact.
		uint64_t newOldNodeOffset = fss.getFreeRegion(node->computeSize());
		tx.seek(newOldNodeOffset);
		node->write(tx);

		// Even though I *think* it will be intact, let's double check.
		CHECK_INTERRUPTION(3, false);

		// Link old node as child of new parent
		newParent = newParent->addChild(childIndex, newOldNodeOffset);

		if (keyIndex + mismatchAt == key.size()) {
			// The parent already matches our key - the parent value should be updated
			newParent->dataOffset = value;
		} else {
			// We need a new leaf as the key is not a substring of the original prefix

			// Create new leaf for the inserted key
			uint16_t newLeafRemainder = key.size() - (keyIndex + mismatchAt) - 1;
			uint64_t newLeafOffset = createLeaf(tx, nodeOffset, &key[keyIndex + mismatchAt + 1], newLeafRemainder, value);

			// Add the child to the new parent
			newParent = newParent->addChild(key[keyIndex + mismatchAt], newLeafOffset);
		}

		// Write the parent node to the original node offset
		// This way it takes on the parent of the original node
		tx.seek(nodeOffset);
		newParent->write(tx);

		CHECK_INTERRUPTION(4, false);

		free(newParent);
		free(node);

		return false;
	}
}

uint64_t DiskBytestringSearchTree::find(const Bytestring& key) {
	uint64_t nodeOffset;
	uint8_t nodeIndex;
	uint32_t keyIndex;
	disk_bytestring_node_t* node;

	MatchType matchType = findNearestNode(key, nodeOffset, nodeIndex, keyIndex, node);

	if (matchType != MatchType::EXACT)
		return 0;

	return node->dataOffset;
}

uint64_t DiskBytestringSearchTree::remove(const Bytestring& key) {
	uint64_t nodeOffset;
	uint8_t nodeIndex;
	uint32_t keyIndex;
	disk_bytestring_node_t* node;

	MatchType matchType = findNearestNode(key, nodeOffset, nodeIndex, keyIndex, node);

	if (matchType != MatchType::EXACT)
		return 0;

	uint64_t previous = node->dataOffset;

	if (previous != 0) {
		node->dataOffset = 0;

		// Just update the header
		FdTransaction tx(fss.getFile());
		tx.seek(nodeOffset);
		tx.write(*node);

		CHECK_INTERRUPTION(5, 0);
	}

	return previous;
}

void DiskBytestringSearchTree::flush() {
	fss.getFile().flush();
}


DiskBytestringSearchTree::MatchType DiskBytestringSearchTree::findNearestNode(const Bytestring& key, uint64_t& offset, uint8_t& nodeIndex, uint32_t& keyIndex, disk_bytestring_node_t*& node) {
	keyIndex = 0;

	if (key.size() == 0) {
		offset = 0;
		return MatchType::INVALID_KEY;
	}

	FdTransaction reader(fss.getFile());
	offset = rootOffset;
	while (true) {
		// Read node header
		reader.seek(offset);
		node = disk_bytestring_node_t::read(reader);

		// Check if key matches prefix at current position
		if (key.size() < keyIndex + node->prefixLength)
			return MatchType::TO_SPLIT;

		if (memcmp(&key[keyIndex], node->getPrefix(), node->prefixLength) != 0)
			return MatchType::TO_SPLIT;

		keyIndex += node->prefixLength;

		if (keyIndex == key.size())
			return MatchType::EXACT;

		// Key continues: look up next byte in child table
		nodeIndex = key[keyIndex++];
		uint64_t childOffset = node->findChildByByte(nodeIndex);

		if (childOffset == 0)
			// No child exists for this byte, entry ready
			return MatchType::EMPTY_ENTRY;

		// Continue traversal
		offset = childOffset;
	}
}


uint64_t DiskBytestringSearchTree::createLeaf(FdTransaction& tx, uint64_t parent, const uint8_t* prefix, uint16_t prefixLength, uint64_t value) {
	uint32_t nodeMemory = sizeof(disk_bytestring_node_t) + prefixLength + 32;
disk_bytestring_node_t newNode {parent, value, nodeMemory, prefixLength, 0};

	uint64_t offset = fss.getFreeRegion(nodeMemory);
	tx.seek(offset);
	tx.write(newNode);
	if (prefixLength > 0)
		tx.write(prefix, prefixLength);

	return offset;
}
