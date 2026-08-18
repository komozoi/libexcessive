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
#include "ds/ArrayList.h"
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
	ArrayList<Bytestring> strings;

	std::minstd_rand rng(1234);
	std::uniform_real_distribution<double> lengthDistLog(0.0, 20.0);
	uint64_t fastEntropy = 0xfeedbeefc0ffe3;

	for (int i = 0; i < count; i++) {
		size_t len = (size_t)pow(2, lengthDistLog(rng));
		ArrayList<uint8_t> buffer;
		buffer.resize((int)len);
		for (int j = 0; j < (int)len; ++j) {
			buffer.get(j) = (uint8_t)(fastEntropy & 0xFF);
			fastEntropy = ((fastEntropy >> 13) ^ (fastEntropy * 991)) + 7;
		}

		strings.add(Bytestring(buffer.getMemory(), (size_t)buffer.size()));
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

static void writeRaw(const char* path, const uint8_t* data, size_t n) {
	FdHandle sf = FdHandle::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
	FdTransaction tx(sf);
	if (n > 0)
		tx.write(data, n);
	sf.flush();
}

TEST(BytestringTest, ReadEmptyFileIsEmpty) {
	const char* path = TEST_FILE_PATH "ReadEmptyFileIsEmpty.bin";
	writeRaw(path, nullptr, 0);

	FdHandle sf = FdHandle::open(path, O_RDWR);
	FdTransaction tx(sf);
	Bytestring bs(tx);

	EXPECT_EQ(bs.size(), (size_t)0);
	EXPECT_FALSE(bs);
}

TEST(BytestringTest, ReadTruncatedTwoByteLengthIsEmpty) {
	const char* path = TEST_FILE_PATH "ReadTruncatedTwoByteLengthIsEmpty.bin";
	uint8_t data[] = { 0x80 };
	writeRaw(path, data, sizeof(data));

	FdHandle sf = FdHandle::open(path, O_RDWR);
	FdTransaction tx(sf);
	Bytestring bs(tx);

	EXPECT_EQ(bs.size(), (size_t)0);
	EXPECT_FALSE(bs);
}

TEST(BytestringTest, ReadTruncatedThreeByteLengthIsEmpty) {
	const char* path = TEST_FILE_PATH "ReadTruncatedThreeByteLengthIsEmpty.bin";
	uint8_t data[] = { 0xFE, 0x00 };
	writeRaw(path, data, sizeof(data));

	FdHandle sf = FdHandle::open(path, O_RDWR);
	FdTransaction tx(sf);
	Bytestring bs(tx);

	EXPECT_EQ(bs.size(), (size_t)0);
	EXPECT_FALSE(bs);
}

TEST(BytestringTest, ReadTruncatedFourByteLengthIsEmpty) {
	const char* path = TEST_FILE_PATH "ReadTruncatedFourByteLengthIsEmpty.bin";
	uint8_t data[] = { 0xFF };
	writeRaw(path, data, sizeof(data));

	FdHandle sf = FdHandle::open(path, O_RDWR);
	FdTransaction tx(sf);
	Bytestring bs(tx);

	EXPECT_EQ(bs.size(), (size_t)0);
	EXPECT_FALSE(bs);
}

TEST(BytestringTest, ReadEncodedEmptyString) {
	const char* path = TEST_FILE_PATH "ReadEncodedEmptyString.bin";
	{
		FdHandle sf = FdHandle::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
		FdTransaction tx(sf);
		Bytestring empty((void*)"", (size_t)0);
		ASSERT_GT(empty.write(tx), (size_t)0);
		sf.flush();
	}

	FdHandle sf = FdHandle::open(path, O_RDWR);
	FdTransaction tx(sf);
	Bytestring bs(tx);
	EXPECT_EQ(bs.size(), (size_t)0);
}

TEST(BytestringTest, ReadTwoByteLengthPrefix) {
	const char* path = TEST_FILE_PATH "ReadTwoByteLengthPrefix.bin";
	uint8_t payload[200];
	for (int i = 0; i < 200; i++)
		payload[i] = (uint8_t)(i * 3);

	{
		FdHandle sf = FdHandle::open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
		FdTransaction tx(sf);
		Bytestring bs(payload, sizeof(payload));
		ASSERT_GT(bs.write(tx), (size_t)0);
		sf.flush();
	}

	FdHandle sf = FdHandle::open(path, O_RDWR);
	FdTransaction tx(sf);
	Bytestring bs(tx);
	Bytestring expected(payload, sizeof(payload));
	EXPECT_EQ(bs, expected);
}
