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

#ifndef RELIABLEKEYVALUESTORE_BYTESTRING_H
#define RELIABLEKEYVALUESTORE_BYTESTRING_H

#include "stdlib.h"
#include "fs/FdHandle.h"


class Bytestring {
public:
	Bytestring(const char* s);
	Bytestring(void* buffer, size_t length);
	Bytestring(FdTransaction& reader);

	Bytestring(const Bytestring& other);
	Bytestring(Bytestring&& other) noexcept;

	// Allows us to create uninitialized arrays of bytestrings
	// Otherwise I would not include this constructor
	inline Bytestring() : memory(nullptr), length(0) {}

	Bytestring& operator =(Bytestring&& other) noexcept;
	Bytestring& operator =(const Bytestring& other);

	size_t write(FdTransaction& writer) const;
	size_t write(uint8_t* buffer) const;

	bool operator >(const Bytestring& other) const;
	bool operator ==(const Bytestring& other) const;

	inline bool operator <(const Bytestring& other) const {
		return other > *this;
	}

	inline bool operator >=(const Bytestring& other) const {
		return !(other > *this);
	}

	inline bool operator <=(const Bytestring& other) const {
		return !(*this > other);
	}

	inline bool operator !=(const Bytestring& other) const {
		return !(*this == other);
	}

	inline uint8_t& operator[](unsigned int index) {
		return (uint8_t&)memory[index];
	}

	inline const uint8_t& operator[](unsigned int index) const {
		return (uint8_t&)memory[index];
	}

	inline explicit operator bool() const {
		return memory != nullptr;
	}

	// Currently this class uses heap allocations.  In the future,
	// something like a slab or even just stack allocations
	// should be used.
	void* allocate(size_t size) {
		return malloc(size);
	}

	~Bytestring();

	size_t size() const {
		return (size_t)length;
	}

private:
	const char* memory;

	// 2GB is far more than will ever be needed.
	int length;
};


#endif //RELIABLEKEYVALUESTORE_BYTESTRING_H
