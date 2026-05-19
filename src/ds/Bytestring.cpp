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

#include "ds/Bytestring.h"

#include "string.h"


Bytestring::Bytestring(const char* s) : length(strlen(s)) {
	// We don't allocate space for the terminator, since we don't need it.
	memory = (const char*)allocate(length);
	memcpy((void*)memory, s, length);
}

Bytestring::Bytestring(void* buffer, size_t length) : length(length) {
	memory = (const char*)allocate(length);
	memcpy((void*)memory, buffer, length);
}

Bytestring::Bytestring(FdTransaction& reader) {
	// Read a big chunk first to minimize later syscalls
	// Most bytestrings fit within 128 bytes, including length data.
	// We can rewind if we read too much.
	uint8_t buffer[128];
	uint8_t* startPtr;
	ssize_t bytesRead = reader.read(buffer, 128);

	// Get the length
	// This is encoded to minimize disk usage on length data, especially for small bytestrings
	if (*buffer < 128) {
		length = *buffer;
		startPtr = &buffer[1];
	} else if (*buffer < 254) {
		length = (((int)*buffer - 128) << 8) | buffer[1];
		startPtr = &buffer[2];
	} else {
		int numLengthBytes = *buffer - 252;
		length = (((int)buffer[1]) << 8) | buffer[2];
		if (numLengthBytes == 3)
			length = (length << 8) | buffer[3];
		startPtr = &buffer[1 + numLengthBytes];
	}

	// Allocate memory and copy the data into our buffer
	memory = (const char*)allocate(length);
	int bytesLeftInBuffer = &buffer[bytesRead] - startPtr;
	if (bytesLeftInBuffer >= length)
		memcpy((void*)memory, startPtr, length);
	else {
		// Didn't fit into our initial buffer - read the rest
		memcpy((void*)memory, startPtr, bytesLeftInBuffer);
		reader.read((void*)&memory[bytesLeftInBuffer], length - bytesLeftInBuffer);
	}

	// Rewind if needed.
	if (bytesLeftInBuffer > length)
		reader.seek(length - bytesLeftInBuffer, SEEK_CUR);
}

Bytestring::Bytestring(const Bytestring& other) : length(other.length) {
	if (other.memory) {
		memory = (const char *) allocate(length);
		memcpy((void *) memory, (void *) other.memory, length);
	} else
		memory = nullptr;
}

Bytestring::Bytestring(Bytestring&& other) noexcept : memory(other.memory), length(other.length) {
	other.memory = nullptr;
	other.length = 0;
}

Bytestring& Bytestring::operator=(Bytestring&& other) noexcept {
	memory = other.memory;
	length = other.length;

	other.memory = nullptr;
	other.length = 0;

	return *this;
}

Bytestring& Bytestring::operator=(const Bytestring& other) {
	if (&other == this)
		return *this;

	free((void*)memory);

	length = other.length;
	if (other.memory) {
		memory = (const char *) allocate(length);
		memcpy((void *) memory, (void *) other.memory, length);
	} else
		memory = nullptr;

	return *this;
}

size_t Bytestring::write(FdTransaction& writer) const {
	uint8_t lengthBuffer[4];
	int lengthOfLength;

	// Efficiently encode length information
	if (length < 128) {
		*lengthBuffer = (uint8_t)length;
		lengthOfLength = 1;
	} else if (length < 32256) {
		lengthBuffer[0] = 128 + (uint8_t)(length >> 8);
		lengthBuffer[1] = (uint8_t)(length & 255);
		lengthOfLength = 2;
	} else if (length < 65536) {
		lengthBuffer[0] = 254;
		lengthBuffer[1] = (uint8_t)(length >> 8);
		lengthBuffer[2] = (uint8_t)(length & 255);
		lengthOfLength = 3;
	} else {
		lengthBuffer[0] = 255;
		lengthBuffer[1] = (uint8_t)(length >> 16);
		lengthBuffer[2] = (uint8_t)((length >> 8) & 255);
		lengthBuffer[3] = (uint8_t)(length & 255);
		lengthOfLength = 4;
	}

	size_t bytesWritten = 0;
	bytesWritten += writer.write(&lengthBuffer, lengthOfLength);
	bytesWritten += writer.write(memory, length);

	return bytesWritten;
}

size_t Bytestring::write(uint8_t* buffer) const {
	uint8_t* lengthBuffer = buffer;
	int lengthOfLength;

	// Efficiently encode length information
	if (length < 128) {
		*lengthBuffer = (uint8_t)length;
		lengthOfLength = 1;
	} else if (length < 32256) {
		lengthBuffer[0] = 128 + (uint8_t)(length >> 8);
		lengthBuffer[1] = (uint8_t)(length & 255);
		lengthOfLength = 2;
	} else if (length < 65536) {
		lengthBuffer[0] = 254;
		lengthBuffer[1] = (uint8_t)(length >> 8);
		lengthBuffer[2] = (uint8_t)(length & 255);
		lengthOfLength = 3;
	} else {
		lengthBuffer[0] = 255;
		lengthBuffer[1] = (uint8_t)(length >> 16);
		lengthBuffer[2] = (uint8_t)((length >> 8) & 255);
		lengthBuffer[3] = (uint8_t)(length & 255);
		lengthOfLength = 4;
	}

	memcpy(&buffer[lengthOfLength], memory, length);
	return length + lengthOfLength;
}

bool Bytestring::operator>(const Bytestring& other) const {
	// We can alter this function to, for example, sort alphabetically while correctly considering case.

	int minLength = other.length > length ? length : other.length;
	int memcmpResult = memcmp(memory, other.memory, minLength);

	// If the initial memory is the same, then go by length.
	if (memcmpResult == 0)
		return length > other.length;

	return memcmpResult > 0;
}

bool Bytestring::operator==(const Bytestring &other) const {
	if (other.length != length)
		return false;

	return memcmp(memory, other.memory, length) == 0;
}

Bytestring::~Bytestring() {
	free((void*)memory);
}
