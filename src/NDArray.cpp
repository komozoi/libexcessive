
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, except software not created by me.
// If you find this software, please notify me immediately: komozoi@protonmail.com
//
//

#include "../include/NDArray.h"

#include <utility>


NDArray::NDArray(ArrayList<int> shape, NDArrayType type) : shape(std::move(shape)), type(type) {
	initialize();
}

NDArray::NDArray(ArrayList<float> vector) : shape({vector.size()}), type(F32) {
	initialize();
	memcpy(float32, vector.getMemory(), memorySize);
}

NDArray::NDArray(ArrayList<uint8_t> vector) : shape({vector.size()}), type(UINT8) {
	initialize();
	memcpy(uint8, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int>& shape, ArrayList<float> vector) : shape(shape), type(F32) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	memcpy(float32, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int>& shape, ArrayList<uint8_t> vector) : shape(shape), type(UINT8) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	memcpy(uint8, vector.getMemory(), memorySize);
}


size_t NDArray::initialize() {
	size_t totalElements = 1;
	for (int i = 0; i < shape.size(); ++i)
		totalElements *= shape.get(i);

	switch (type) {
		case BINARY:
			memorySize = ((totalElements + 63) / 8) & ~7;
			break;
		case UINT8:
			memorySize = totalElements;
			break;
		case F32:
			memorySize = totalElements * 4;
			break;
		/*case BIGINT:
			// Stored as multiple overlays, each with int64 for value and uint16 for exponent on each element
			// Total size per element is 10 * numOverlays bytes.  Another uint64 is added for metadata like the number
			// of overlays.
			memorySize = 8 + totalElements * 10;
			break;*/
		case UINT256:
			memorySize = totalElements * 32;
			break;
	}

	memory = malloc(memorySize);
	bzero(memory, memorySize);
	//if (type == BIGINT)
	//	*uint64 = 1;

	return totalElements;
}


bool NDArray::operator==(const NDArray& other) const {
	if (type != other.type)
		return false;
	if (shape.size() != other.shape.size())
		return false;
	for (int i = 0; i < shape.size(); ++i)
		if (shape.get(i) != other.shape.get(i))
			return false;

	// It is OK to always compare uint64 for the first batch here because data should be identical
	// regardless of encoding
	size_t i = 0;
	for (; i < memorySize - 7; i += 8)
		if (uint64[i / 8] != other.uint64[i / 8])
			return false;

	// Finish up with byte comparison
	for (; i < memorySize; ++i)
		if (uint8[i] != other.uint8[i])
			return false;

	return true;
}


NDArray::~NDArray() {
	free(memory);
}


size_t NDArray::computeOffset(const ArrayList<int>& indices) const  {
	size_t offset = 0;
	size_t stride = 1;
	// walk from the last dimension
	for (int d = shape.size() - 1; d >= 0; --d) {
		int idx = indices.get(d);
		if (idx < 0 || idx >= shape.get(d))
			throw std::out_of_range("index out of bounds");
		offset += idx * stride;
		stride *= shape.get(d);
	}
	return offset;
}
