/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-4-6
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

#ifndef LIBEXCESSIVE_HASH_H
#define LIBEXCESSIVE_HASH_H

#include <functional>


size_t excessiveFastHash(const void* data, size_t size);


// Default hash method
// This gets around std::hash<K>{}(value) for types that don't have a custom hash function (like c strings)

template<typename T>
size_t obviousHashFunction(const T& value) {
	return std::hash<T>{}(value);
}

size_t obviousHashFunction(char* value);
size_t obviousHashFunction(const char* value);


template<typename T>
int obviousCompareFunction(const T& a, const T& b) {
	return a == b ? 0 : 0x7FFFFFFF;
}

int obviousCompareFunction(const char* a, const char* b);

#endif //LIBEXCESSIVE_HASH_H