/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-10-11
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


#ifndef INTELLIGENCELIB_FREESPACEFILE_H
#define INTELLIGENCELIB_FREESPACEFILE_H

#include "FdHandle.h"
#include "BTree.h"


/**
 * This is a base to build other formats on.
 * It keeps a BTree of free regions in the file.
 */


class FreeSpaceFile {
public:
	explicit FreeSpaceFile(FdHandle&& file);
	explicit FreeSpaceFile(const FdHandle& file);

	void markFreeRegion(off_t start, uint32_t length);

	off_t getFreeRegion(uint32_t length);

	off_t getHeaderEnd();

protected:
	struct free_region_pair_t {
		uint64_t offset;
		uint32_t size;

		static int compare(const free_region_pair_t& a, const free_region_pair_t& b) {
			// This sorts by size first, ascending, then by offset, descending.
			// This way the BTree will return the next-largest element with the lowest offset first
			// This also prevents the BTree from treating two different entries of the same
			// size as identical by sorting the offset too.
			if (a.size > b.size)
				return 1;
			else if (a.size < b.size)
				return -1;
			else if (a.offset > b.offset)
				return 1;
			else if (a.offset < b.offset)
				return -1;
			else
				return 0;
		}
	};

	FdHandle file;
	BTree<free_region_pair_t> freeRegions;
};


#endif //INTELLIGENCELIB_FREESPACEFILE_H
