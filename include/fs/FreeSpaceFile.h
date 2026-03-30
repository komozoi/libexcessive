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
 * @brief Base class for files that manage free space using a B-Tree.
 * 
 * This class keeps a B-Tree of free regions in the file to allow efficient
 * allocation and deallocation of space.
 */
class FreeSpaceFile {
public:
	/**
	 * @brief Constructs a FreeSpaceFile from an FdHandle (move).
	 * @param file The file handle.
	 */
	explicit FreeSpaceFile(FdHandle&& file);

	/**
	 * @brief Constructs a FreeSpaceFile from an FdHandle (copy).
	 * @param file The file handle.
	 */
	explicit FreeSpaceFile(const FdHandle& file);

	/**
	 * @brief Marks a region of the file as free.
	 * @param start Starting offset of the region.
	 * @param length Length of the region in bytes.
	 */
	void markFreeRegion(off_t start, uint32_t length);

	/**
	 * @brief Finds and allocates a free region of at least the specified length.
	 * @param length Minimum required length.
	 * @return Offset of the allocated region, or -1 if no suitable region is found.
	 */
	off_t getFreeRegion(uint32_t length);

	/**
	 * @brief Gets the offset immediately following the file header.
	 * @return End offset of the header.
	 */
	off_t getHeaderEnd();

protected:
	/**
	 * @brief Represents a free region in the file.
	 */
	struct free_region_pair_t {
		uint64_t offset; /**< Starting offset of the free region. */
		uint32_t size;   /**< Size of the free region in bytes. */

		/**
		 * @brief Comparison function for free regions.
		 * 
		 * Sorts by size ascending, then by offset ascending.
		 * 
		 * @param a First region.
		 * @param b Second region.
		 * @return Comparison result.
		 */
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

	FdHandle file;                        /**< The file handle. */
	BTree<free_region_pair_t> freeRegions; /**< B-Tree of free regions. */
};


#endif //INTELLIGENCELIB_FREESPACEFILE_H
