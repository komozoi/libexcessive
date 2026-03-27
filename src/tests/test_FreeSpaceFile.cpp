/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-02-26
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


#include <string>
#include <cstdint>
#include "universaltime.h"
#include "gtest/gtest.h"
#include "fs/FdHandle.h"
#include "fs/FreeSpaceFile.h"
#include "fcntl.h"
#include "ds/ArrayList.h"


static std::string makeTempFileName(const char* base) {
	char buf[256];
	uint64_t t = millis_since_epoch();
	snprintf(buf, sizeof(buf), "%s_%llu.tmp", base, (unsigned long long)t);
	return std::string(buf);
}


TEST(FreeSpaceFileTest, BasicAllocateFree) {
	std::string path = makeTempFileName("freespace_basic");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);

	FreeSpaceFile fs(std::move(handle));

	off_t headerEnd = fs.getHeaderEnd();

	fs.markFreeRegion(headerEnd, 1024);

	off_t region = fs.getFreeRegion(512);

	EXPECT_GE(region, headerEnd);
	EXPECT_NE(region, (off_t)-1);

	unlink(test_file);
}

TEST(FreeSpaceFileTest, ExactMatchConsumed) {
	std::string path = makeTempFileName("freespace_exact");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(std::move(handle));

	off_t r1 = fs.getFreeRegion(4096);
	EXPECT_NE(r1, (off_t)-1);

	fs.markFreeRegion(r1, 4096);

	off_t r2 = fs.getFreeRegion(4096);
	EXPECT_EQ(r2, r1);

	off_t r3 = fs.getFreeRegion(1);
	EXPECT_NE(r3, r1); // should not reuse again

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, RegionReuseWithoutSplitting) {
	std::string path = makeTempFileName("freespace_split");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(std::move(handle));

	off_t base = fs.getFreeRegion(4096);

	fs.markFreeRegion(base, 4096);

	off_t r1 = fs.getFreeRegion(1024);
	EXPECT_EQ(r1, base);

	off_t r2 = fs.getFreeRegion(1024);
	EXPECT_GE(r2, base + 4096);

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, SmallestSufficientChosen) {
	std::string path = makeTempFileName("freespace_bestfit");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(std::move(handle));

	off_t a = fs.getFreeRegion(4096);
	off_t b = fs.getFreeRegion(1024);
	off_t c = fs.getFreeRegion(2048);

	fs.markFreeRegion(a, 4096);
	fs.markFreeRegion(b, 1024);
	fs.markFreeRegion(c, 2048);

	off_t r = fs.getFreeRegion(1000);

	EXPECT_EQ(r, b);

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, EqualSizeLowestOffsetFirst) {
	std::string path = makeTempFileName("freespace_equal");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(std::move(handle));

	off_t r1 = fs.getFreeRegion(2048);
	off_t r2 = fs.getFreeRegion(2048);
	off_t r3 = fs.getFreeRegion(2048);

	fs.markFreeRegion(r2, 2048);
	fs.markFreeRegion(r1, 2048);
	fs.markFreeRegion(r3, 2048);

	off_t g1 = fs.getFreeRegion(1024);
	off_t g2 = fs.getFreeRegion(1024);

	EXPECT_EQ(g1, std::min(r1, std::min(r2, r3)));
	EXPECT_EQ(g2, std::min(
			(r1 == g1 ? std::max(r2, r3) : r1),
			(r2 == g1 ? std::max(r1, r3) : r2)
	));

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, RequestTooLargeReturnsNewSpace) {
	std::string path = makeTempFileName("freespace_toolarge");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(handle);

	// Create a bunch of regions
	ArrayList<off_t> dummyOffsets(256);
	for (int i = 0; i < 250; i++) {
		uint32_t shuffled = (0xFF & (i * 997)) + 2;
		uint32_t size = shuffled * 16;
		off_t off = fs.getFreeRegion(size);
		handle.seek(off);
		handle.write("| Region start -->", 18);
		handle.seek(off + size - 16);
		handle.write("<-- Region end |", 16);
		dummyOffsets.add(off);
	}

	// Free them all
	for (int i = 0; i < 250; i++) {
		uint32_t shuffled = 0xFF & (i * 997);
		uint32_t size = shuffled * 16;
		fs.markFreeRegion(dummyOffsets.get(i), size);
	}

	off_t lastAllocatedLocation = dummyOffsets.get(dummyOffsets.size() - 1);

	off_t large = fs.getFreeRegion(4096);

	EXPECT_GT(large, lastAllocatedLocation);

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, ReturnsCorrectLocation) {
	std::string path = makeTempFileName("freespace_corretlocation");
	const char* test_file = path.c_str();

	off_t correctOffset = 0;
	uint32_t bestSize = 9000;

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(handle);

	// Create a bunch of regions
	ArrayList<off_t> dummyOffsets(256);
	for (int i = 0; i < 250; i++) {
		uint32_t shuffled = (0xFF & (i * 997)) + 2;
		uint32_t size = shuffled * 128;
		off_t off = fs.getFreeRegion(size);
		handle.seek(off);
		handle.write("| Region start -->", 18);
		handle.seek(off + size - 16);
		handle.write("<-- Region end |", 16);
		dummyOffsets.add(off);
	}

	// Free them all
	for (int i = 0; i < 250; i++) {
		uint32_t shuffled = 0xFF & (i * 997);
		uint32_t size = shuffled * 128;
		if (size < bestSize && size >= 4096) {
			bestSize = size;
			correctOffset = dummyOffsets.get(i);
		}
		fs.markFreeRegion(dummyOffsets.get(i), size);
	}

	off_t region = fs.getFreeRegion(4096);

	EXPECT_EQ(region, correctOffset);

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, ManyRegionsStress) {
	std::string path = makeTempFileName("freespace_stress");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(std::move(handle));

	const int count = 500;

	std::vector<off_t> blocks;

	for (int i = 0; i < count; i++) {
		blocks.push_back(fs.getFreeRegion(4096));
	}

	for (int i = 0; i < count; i++) {
		fs.markFreeRegion(blocks[i], 4096);
	}

	for (int i = 0; i < count; i++) {
		off_t r = fs.getFreeRegion(4096);
		EXPECT_NE(r, (off_t)-1);
	}

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, ReinsertAfterAllocate) {
	std::string path = makeTempFileName("freespace_reinsert");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(std::move(handle));

	off_t base = fs.getFreeRegion(2048);

	fs.markFreeRegion(base, 2048);

	off_t r1 = fs.getFreeRegion(1024);
	EXPECT_EQ(r1, base);

	fs.markFreeRegion(r1, 1024);

	off_t r2 = fs.getFreeRegion(1024);
	EXPECT_EQ(r2, base);

	if (!HasFailure()) unlink(test_file);
}

TEST(FreeSpaceFileTest, NeverAllocatesBeforeHeader) {
	std::string path = makeTempFileName("freespace_header");
	const char* test_file = path.c_str();

	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	FreeSpaceFile fs(std::move(handle));

	off_t headerEnd = fs.getHeaderEnd();

	off_t r = fs.getFreeRegion(128);

	EXPECT_GE(r, headerEnd);

	if (!HasFailure()) unlink(test_file);
}




