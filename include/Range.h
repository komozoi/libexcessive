/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-02-19
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


#ifndef EXCESSIVE_RANGE_H
#define EXCESSIVE_RANGE_H

#include <utility>


template <class T>
class Range {
public:
	Range(T&& begin, T&& end) : a(std::move(begin)), b(std::move(end)) {}
	Range(const T& begin, const T& end) : a(begin), b(end) {}

	T begin() { return a; }
	T end() { return b; }

	const T& begin() const { return a; }
	const T& end() const { return b; }

private:
	T a;
	T b;
};


#endif //EXCESSIVE_RANGE_H
