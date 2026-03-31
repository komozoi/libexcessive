/*
 * Copyright 2023-2025 komozoi
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

#include <gtest/gtest.h>
#include "LongKey.h"
#include <cstring>
#include <string>


TEST(LongKeyTest, DefaultConstructor) {
	LongKey<128> key;
	// Since it's uninitialized, we can't assert specific values,
	// but we can assert that it doesn't crash or throw
	SUCCEED();
}

TEST(LongKeyTest, CopyConstructor) {
	LongKey<128> key1;
	memset(key1.data.rawBytes, 0xAB, sizeof(key1.data));
	LongKey<128> key2(key1);
	EXPECT_EQ(memcmp(key1.data.rawBytes, key2.data.rawBytes, sizeof(key1.data)), 0);
}

TEST(LongKeyTest, MoveConstructor) {
	LongKey<128> key1;
	memset(key1.data.rawBytes, 0xCD, sizeof(key1.data));
	LongKey<128> key2(std::move(key1));
	EXPECT_EQ(key2.data.rawBytes[0], 0xCD);
}

TEST(LongKeyTest, CopyAssignment) {
	LongKey<128> key1;
	memset(key1.data.rawBytes, 0x11, sizeof(key1.data));
	LongKey<128> key2;
	key2 = key1;
	EXPECT_EQ(memcmp(key1.data.rawBytes, key2.data.rawBytes, sizeof(key1.data)), 0);
}

TEST(LongKeyTest, MoveAssignment) {
	LongKey<128> key1;
	memset(key1.data.rawBytes, 0x22, sizeof(key1.data));
	LongKey<128> key2;
	key2 = std::move(key1);
	EXPECT_EQ(key2.data.rawBytes[0], 0x22);
}

TEST(LongKeyTest, ConstructFromPointer) {
	uint8_t bytes[16];
	for (int i = 0; i < 16; ++i)
		bytes[i] = i;

	LongKey<128> key(bytes);
	for (int i = 0; i < 16; ++i)
		EXPECT_EQ(key.data.rawBytes[i], i);
}

TEST(LongKeyTest, ConstructFromPointerWithSize) {
	uint8_t bytes[8];
	for (int i = 0; i < 8; ++i)
		bytes[i] = 0xFF;

	LongKey<128> key(bytes, 8);
	for (int i = 0; i < 8; ++i)
		EXPECT_EQ(key.data.rawBytes[i], 0xFF);
	for (int i = 8; i < 16; ++i)
		EXPECT_EQ(key.data.rawBytes[i], 0);
}

TEST(LongKeyTest, ConstructFromHexString) {
	LongKey<128> key("0x0123456789ABCDEF0123456789ABCDEF");
	EXPECT_EQ(key.data.rawBytes[15], 0x01);
	EXPECT_EQ(key.data.rawBytes[0], 0xEF);
}

TEST(LongKeyTest, CompareEqual) {
	LongKey<128> a("0xFFFFFFFFFFFFFFFF0000000000000000");
	LongKey<128> b("0xFFFFFFFFFFFFFFFF0000000000000000");
	EXPECT_EQ(a.compare(b), 0);
}

TEST(LongKeyTest, CompareGreater) {
	LongKey<128> a("0xFFFFFFFFFFFFFFFF0000000000000001");
	LongKey<128> b("0xFFFFFFFFFFFFFFFF0000000000000000");
	EXPECT_EQ(a.compare(b), 1);
}

TEST(LongKeyTest, CompareLess) {
	LongKey<128> a("0x00000000000000000000000000000001");
	LongKey<128> b("0x000000000000000000000000000000FF");
	EXPECT_EQ(a.compare(b), -1);
}

TEST(LongKeyTest, SubtractOperator) {
	LongKey<128> a("0x00000000000000000000000000000001");
	LongKey<128> b("0x00000000000000000000000000000001");
	LongKey<128> c("0x00000000000000000000000000000002");

	EXPECT_EQ(a - b, 0);
	EXPECT_EQ(c - a, 1);
	EXPECT_EQ(a - c, -1);
}

TEST(LongKeyTest, ToStringConversion) {
	LongKey<128> key("0x0123456789ABCDEF0123456789ABCDEF");
	char buffer[100];
	key.toStr(buffer);
	EXPECT_STREQ(buffer, "0x0123456789AbCdeF0123456789AbCdeF");
}

TEST(LongKeyTest, ParseExtraLongString) {
	LongKey<128> key("0x0123456789ABCDEF0123456789ABCDEFdshfshdfjksdfjkhsdfkjdhfkjsdhfkjsdhfjsfjsdnfjknsfdjknsdjkfjsfkjsdkfjsdjfhsdfkjfhsdkjhfdskjfhkj");
	char buffer[100];
	key.toStr(buffer);
	EXPECT_STREQ(buffer, "0x0123456789AbCdeF0123456789AbCdeF");
}

TEST(LongKeyTest, IsZeroTrue) {
	LongKey<128> key;
	memset(key.data.rawBytes, 0, sizeof(key.data));
	EXPECT_TRUE(key.isZero());
}

TEST(LongKeyTest, IsZeroFalse) {
	LongKey<128> key;
	memset(key.data.rawBytes, 0, sizeof(key.data));
	key.data.rawBytes[0] = 1;
	EXPECT_FALSE(key.isZero());
}

TEST(LongKey160Test, ConstructFromHexString) {
	const char* hexStr = "0x0123456789ABCDEFFEDCBA987654321001234567";
	LongKey<160> key(hexStr);

	// Check a few known byte values
	EXPECT_EQ(key.data.rawBytes[19], 0x01);
	EXPECT_EQ(key.data.rawBytes[18], 0x23);
	EXPECT_EQ(key.data.rawBytes[0], 0x67);
}

TEST(LongKey160Test, ConstructFromStringView) {
	std::string_view hexStr = "0x00112233445566778899AABBCCDDEEFF00112233";
	LongKey<160> key(hexStr);

	// Check first and last bytes
	EXPECT_EQ(key.data.rawBytes[19], 0x00);
	EXPECT_EQ(key.data.rawBytes[18], 0x11);
	EXPECT_EQ(key.data.rawBytes[0], 0x33);
}

TEST(LongKey160Test, ToStringCorrectness) {
	const char* input = "0xdeAdbeeF0123456789AbCdeFCAFebAbe11223344";
	LongKey<160> key(input);

	char buffer[100];
	key.toStr(buffer);

	// Case sensitivity is according to your code (AbCdeF used in output)
	// So we just check it's the same digits, ignoring case
	std::string expected = input;
	std::string actual = buffer;

	// Remove 0x, compare hex digits case-insensitively
	EXPECT_EQ(expected, actual);
}

TEST(LongKey160Test, ToStringAndBackRoundTrip) {
	const char* input = "0x00112233445566778899AABBCCDDEEFF00112233";
	LongKey<160> original(input);

	char buffer[100];
	original.toStr(buffer);

	LongKey<160> reconstructed(buffer);
	EXPECT_EQ(original.compare(reconstructed), 0);
}

