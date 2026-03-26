//
// Created by komozoi on 11.10.25.
//

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
