/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-02-08
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


#ifndef EXCESSIVE_STRUTIL_H
#define EXCESSIVE_STRUTIL_H

#include <string_view>
#include <cctype>
#include <cstdarg>
#include "stdint.h"
#include "stdio.h"
#include "string.h"


static inline bool isNumber(const std::string_view& s) {
	std::string_view::const_iterator it = s.begin();
	while (it != s.end() && std::isdigit(*it)) ++it;
	return !s.empty() && it == s.end();
}


/**
 * Finds the end of the string and adds the formatted string to it while formatting.
 * @param s1 first string to concat, usually output from a previous concatf() call
 * @param s2 a format string
 * @return
 */
static inline char* concatf(char* s1, const char* s2, ...) {
	while (*s1)
		s1++;

	va_list args;
	va_start(args, s2);
	s1 = &s1[vsprintf(s1, s2, args)];
	va_end(args);

	return s1;
}

static inline uint8_t parseHexDigit(char c) {
	if ('0' <= c && '9' >= c)
		return c - '0';
	else if ('a' <= c && 'f' >= c)
		return c - 'a' + 10;
	else if ('A' <= c && 'F' >= c)
		return c - 'A' + 10;
	return 255;
}


static inline uint32_t parseHexStr32(std::string_view s) {
	if (s.size() > 1 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
		s = s.substr(2);

	uint32_t out = 0;
	for (char i : s)
		out = (out << 4) | parseHexDigit(i);

	return out;
}


static inline uint64_t parseHexStr64(std::string_view s) {
	if (s.size() > 1 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
		s = s.substr(2);

	uint64_t out = 0;
	for (char i : s)
		out = (out << 4) | parseHexDigit(i);

	return out;
}


static inline char* toHex(uint64_t value, char* outBuffer) {
	/*const char* base = "0123456789Abcdef";
	*(outBuffer++) = '0';
	*(outBuffer++) = 'x';

	if (value == 0)
		*(outBuffer++) = '0';
	else {
		while (value) {
			*(outBuffer++) = base[value & 15];
			value >>= 4;
		}
	}

	*(outBuffer) = 0;

	return outBuffer;*/
	return &outBuffer[sprintf(outBuffer, "0x%lx", value)];
}


static inline char* toHex(const uint8_t* data, int size, char* outBuffer) {
	if (data == nullptr)
		return stpcpy(outBuffer, "(null)");

	const char* base = "0123456789abcdef";
	*(outBuffer++) = '0';
	*(outBuffer++) = 'x';

	for (int i = 0; i < size; i++) {
		*(outBuffer++) = base[data[i] >> 4];
		*(outBuffer++) = base[data[i] & 15];
	}

	*(outBuffer) = 0;

	return outBuffer;
}


// Formats similar to "hexdump -C", but the number of columns is adjustable.
// If columns is -1, then only one row is created.
// The returned pointer is allocated with malloc, so be sure to free it.
char* formatBinaryDataForHexdump(const uint8_t* data, int size, int columns=-1);


#endif //EXCESSIVE_STRUTIL_H
