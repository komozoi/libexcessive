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
#include "strutil.h"

TEST(StrUtilTest, Concatf) {
	char buffer[256];
	buffer[0] = '\0';

	// Use (char*) to force the pointer version
	char* end = concatf((char*)buffer, "Hello");
	EXPECT_STREQ(buffer, "Hello");
	EXPECT_EQ(end, buffer + 5);
	EXPECT_EQ(*end, '\0');

	end = concatf(end, " %s", "World");
	EXPECT_STREQ(buffer, "Hello World");
	EXPECT_EQ(end, buffer + 11);
	EXPECT_EQ(*end, '\0');

	end = concatf(end, "!");
	EXPECT_STREQ(buffer, "Hello World!");
	EXPECT_EQ(end, buffer + 12);
	EXPECT_EQ(*end, '\0');
}

TEST(StrUtilTest, ConcatfEmpty) {
	char buffer[64];
	buffer[0] = '\0';
	char* end = concatf((char*)buffer, "");
	EXPECT_STREQ(buffer, "");
	EXPECT_EQ(end, buffer);
}

TEST(StrUtilTest, ConcatfFormatted) {
	char buffer[64];
	buffer[0] = '\0';
	concatf((char*)buffer, "%d + %d = %d", 1, 1, 2);
	EXPECT_STREQ(buffer, "1 + 1 = 2");
}

TEST(StrUtilTest, ConcatfLarge) {
	char buffer[2048];
	buffer[0] = '\0';
	
	std::string large(1500, 'a');
	char* end = concatf((char*)buffer, "%s", large.c_str());
	EXPECT_EQ(strlen(buffer), 1500);
	EXPECT_EQ(end, buffer + 1500);
}
