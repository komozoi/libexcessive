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

#include <gtest/gtest.h>
#include <random>
#include "fcntl.h"

#include "fs/FdHandle.h"
#include "ds/Bytestring.h"
#include "universaltime.h"


#define TEST_FILE_PATH "/tmp/bytestringTest_"


TEST(BytestringTest, ConstructFromCStringAndCompare) {
	Bytestring a("hello");
	Bytestring b("world");
	Bytestring c("hello");

	ASSERT_TRUE(a == c);
	ASSERT_TRUE(a != b);
	ASSERT_TRUE(b > a);
	ASSERT_TRUE(a < b);
}

TEST(BytestringTest, ConstructFromBufferAndCompare) {
	const char buf1[] = {0x01, 0x02, 0x03};
	const char buf2[] = {0x01, 0x02, 0x04};

	Bytestring bs1((void*)buf1, sizeof(buf1));
	Bytestring bs2((void*)buf2, sizeof(buf2));
	Bytestring bs3((void*)buf1, sizeof(buf1));

	ASSERT_TRUE(bs1 == bs3);
	ASSERT_TRUE(bs1 != bs2);
	ASSERT_TRUE(bs2 > bs1);
	ASSERT_TRUE(bs1 <= bs3);
	ASSERT_TRUE(bs2 >= bs1);
}

TEST(BytestringTest, WriteAndReadBackFromFile) {
	const char data[] = "persistent_data";

	// Write the Bytestring to file
	const char* path = TEST_FILE_PATH "WriteAndReadBackFromFile.bin";
	{
		FdHandle sf = FdHandle::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
		FdTransaction tx(sf);
		Bytestring bs((void*)data, sizeof(data));
		ASSERT_GT(bs.write(tx), 0);
		sf.flush();
	}

	// Read the Bytestring back from file
	{
		FdHandle sf = FdHandle::open(path, O_RDWR);
		FdTransaction tx(sf);
		Bytestring bsFromFile(tx);
		Bytestring bsExpected((void*)data, sizeof(data));
		ASSERT_TRUE(bsFromFile == bsExpected);
	}
}

TEST(BytestringTest, WriteAndReadBackFromFileMany) {
	int count = 1000;
	std::vector<Bytestring> strings;

	std::minstd_rand rng(1234);
	std::uniform_real_distribution<double> lengthDistLog(0.0, 20.0);
	uint64_t fastEntropy = 0xfeedbeefc0ffe3;

	for (int i = 0; i < count; i++) {
		size_t len = (size_t)pow(2, lengthDistLog(rng));
		std::vector<uint8_t> buffer(len);
		for (size_t j = 0; j < len; ++j) {
			buffer[j] = fastEntropy & 0xFF;
			fastEntropy = ((fastEntropy >> 13) ^ (fastEntropy * 991)) + 7;
		}

		strings.emplace_back(Bytestring(buffer.data(), buffer.size()));
	}

	const char* path = TEST_FILE_PATH "WriteAndReadBackFromFileMany.bin";
	{
		FdHandle sf = FdHandle::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
		FdTransaction tx(sf);
		uint64_t timer = millis_since_epoch();
		for (Bytestring& s: strings)
			s.write(tx);
		timer = millis_since_epoch() - timer;
		printf("Wrote %i bytestrings in %lums (%.2fus per bytestring)\n", count, timer, (1000.0 * timer) / count);
	}

	{
		FdHandle sf = FdHandle::open(path, O_RDWR, 0644);
		FdTransaction tx(sf);
		uint64_t timer = millis_since_epoch();
		for (Bytestring& expected: strings) {
			Bytestring readBack(tx);
			EXPECT_EQ(expected, readBack);
		}
		timer = millis_since_epoch() - timer;
		printf("Read %i bytestrings in %lums (%.2fus per bytestring)\n", count, timer, (1000.0 * timer) / count);
	}
}

TEST(BytestringTest, MoveAndCopySemantics) {
	Bytestring original("foobar");

	// Test copy constructor
	Bytestring copy = original;
	ASSERT_TRUE(copy == original);

	// Test move constructor
	Bytestring moved = std::move(original);
	ASSERT_TRUE(moved == copy);

	// Test copy assignment
	Bytestring assigned("glupy");
	assigned = moved;
	ASSERT_TRUE(assigned == moved);

	// Test move assignment
	Bytestring moveAssigned("blah");
	moveAssigned = std::move(assigned);
	ASSERT_TRUE(moveAssigned == moved);
}

TEST(BytestringTest, CompareLexicographically) {
	Bytestring a("abc");
	Bytestring b("abcd");
	Bytestring c("abc");

	ASSERT_TRUE(a < b);
	ASSERT_TRUE(b > a);
	ASSERT_TRUE(a == c);
	ASSERT_TRUE(b != c);
}
