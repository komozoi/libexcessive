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


#include "fs/FreeSpaceFile.h"

#include "unistd.h"



FreeSpaceFile::FreeSpaceFile(FdHandle&& _file) : file(std::move(_file)), freeRegions(file, 0, free_region_pair_t::compare) {
}

FreeSpaceFile::FreeSpaceFile(const FdHandle &file) : file(file), freeRegions(file, 0, free_region_pair_t::compare) {
}

void FreeSpaceFile::markFreeRegion(off_t start, uint32_t length) {
	freeRegions.insert({(uint64_t)start, length});
}

off_t FreeSpaceFile::getFreeRegion(uint32_t length) {
	// Attempt to find a space earlier in the file first
	free_region_pair_t pair{0, length};
	if (freeRegions.findNext(pair)) {
		freeRegions.remove(pair);
		return pair.offset;
	}

	// No space found, go to the end of the file instead
	off_t out = file.seekToEndWithPadding(8);
	ftruncate(file.getFd(), out + length);
	return out;
}

off_t FreeSpaceFile::getHeaderEnd() {
	return freeRegions.getHeaderEndOffset();
}
