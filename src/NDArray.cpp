
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, except software not created by me.
// If you find this software, please notify me immediately: komozoi@protonmail.com
//
//

#include "../include/NDArray.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>


/*
** TYPE RULES
**
** Arithmetic never silently loses information.  When two values meet, the
** result type is one that can represent both operands losslessly
** (see promoteTypes).  Explicit convert() is the only place the programmer
** may request a potentially lossy cast.
*/

static bool isFloatingPoint(NDArrayType type) {
	return type == F32;
}

static bool isSigned(NDArrayType type) {
	return type == F32;
}

/**
 * @param type type to inspect
 * @return `n` where `2^n` is the largest power of 2 that can fit in the type
 *         while keeping integer precision
 */
static int maxPowerOfTypeIntPrecision(NDArrayType type) {
	switch (type) {
		case BINARY:
			return 0;
		case UINT8:
			return 7;
		case F32:
			return 24;
		case UINT256:
			return 255;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

/**
 * @param type type to inspect
 * @return `n` where `2^n` is the largest power of 2 that can fit in the type
 */
static int maxPowerOfType(NDArrayType type) {
	switch (type) {
		case BINARY:
			return 0;
		case UINT8:
			return 7;
		case F32:
			return 128;
		case UINT256:
			return 255;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

// Kept for the type-rules framework (range checks / future types).
static void retainTypeRuleHelpers() {
	(void)&maxPowerOfType;
}

static bool isLosslessConversion(NDArrayType from, NDArrayType to) {
	if (from == to)
		return true;
	if (maxPowerOfTypeIntPrecision(to) < maxPowerOfTypeIntPrecision(from))
		return false;
	if (isSigned(from) && !isSigned(to))
		return false;
	return true;
}

/**
 * Type that can hold both `a` and `b` without losing integer precision or
 * signedness.  BINARY+BINARY promotes to UINT8 because 1+1 no longer fits
 * in a single bit.
 */
static NDArrayType promoteTypes(NDArrayType a, NDArrayType b) {
	if (a == b) {
		if (a == BINARY)
			return UINT8;
		return a;
	}
	if (isLosslessConversion(a, b))
		return b;
	if (isLosslessConversion(b, a))
		return a;
	throw std::invalid_argument("NDArray: no lossless common type for operands");
}

// F64 is not yet a supported storage type; double scalars promote like F32.
static NDArrayType promoteWithFloatScalar(NDArrayType arrayType) {
	return promoteTypes(arrayType, F32);
}

// Signed int scalars: F32 is the only signed storage type.  UINT256 stays
// UINT256 (non-negative ints convert exactly into it).
static NDArrayType promoteWithIntScalar(NDArrayType arrayType) {
	if (arrayType == UINT256)
		return UINT256;
	return promoteTypes(arrayType, F32);
}


template<typename A, typename B>
static void addBasic(A* a, B* b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] += (A)b[i];
}

template<typename A, typename B>
static void subBasic(A* a, B* b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] -= (A)b[i];
}

template<typename A, typename B>
static void mulBasic(A* a, B* b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] *= (A)b[i];
}

template<typename A, typename B>
static void divBasic(A* a, B* b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] /= (A)b[i];
}

template<typename A, typename B>
static void addBasic(A* a, B b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] += (A)b;
}

template<typename A, typename B>
static void subBasic(A* a, B b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] -= (A)b;
}

template<typename A, typename B>
static void mulBasic(A* a, B b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] *= (A)b;
}

template<typename A, typename B>
static void divBasic(A* a, B b, size_t size) {
	for (size_t i = 0; i < size; ++i)
		a[i] /= (A)b;
}


// ---- BINARY bit helpers ----------------------------------------------------

static uint8_t binaryGet(const uint64_t* words, size_t i) {
	return (uint8_t)((words[i >> 6] >> (i & 63)) & 1ULL);
}

static void binarySet(uint64_t* words, size_t i, uint8_t bit) {
	if (bit)
		words[i >> 6] |= 1ULL << (i & 63);
	else
		words[i >> 6] &= ~(1ULL << (i & 63));
}


static void requireSameShape(const NDArray& a, const NDArray& b) {
	if (a.shape.size() != b.shape.size())
		throw std::invalid_argument("NDArray: shape rank mismatch");
	for (int i = 0; i < a.shape.size(); ++i)
		if (a.shape.get(i) != b.shape.get(i))
			throw std::invalid_argument("NDArray: shape mismatch");
}

static bool isPrefixShape(const ArrayList<int>& prefix, const ArrayList<int>& full) {
	if (prefix.size() > full.size())
		return false;
	for (int i = 0; i < prefix.size(); ++i)
		if (prefix.get(i) != full.get(i))
			return false;
	return true;
}


// ---- construction ----------------------------------------------------------

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

NDArray::NDArray(const NDArray& other) : shape(other.shape), type(other.type), memorySize(other.memorySize) {
	memory = malloc(memorySize ? memorySize : 1);
	if (memorySize)
		memcpy(memory, other.memory, memorySize);
}

NDArray::NDArray(NDArray&& other) noexcept
	: shape(other.shape),
	  type(other.type),
	  memorySize(other.memorySize) {
	memory = other.memory;
	other.memory = nullptr;
	other.memorySize = 0;
}


size_t NDArray::initialize() {
	size_t totalElements = 1;
	for (int i = 0; i < shape.size(); ++i)
		totalElements *= shape.get(i);

	switch (type) {
		case BINARY:
			// Bytes, rounded up to a whole number of uint64_t words.
			memorySize = ((totalElements + 63) / 64) * 8;
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

	memory = malloc(memorySize ? memorySize : 1);
	if (memorySize)
		bzero(memory, memorySize);
	//if (type == BIGINT)
	//	*uint64 = 1;

	return totalElements;
}


size_t NDArray::numElements() const {
	size_t n = 1;
	for (int i = 0; i < shape.size(); ++i)
		n *= (size_t)shape.get(i);
	return n;
}


bool NDArray::operator==(const NDArray& other) const {
	if (type != other.type)
		return false;
	if (shape.size() != other.shape.size())
		return false;
	for (int i = 0; i < shape.size(); ++i)
		if (shape.get(i) != other.shape.get(i))
			return false;

	if (memorySize == 0)
		return true;

	// It is OK to always compare uint64 for the first batch here because data should be identical
	// regardless of encoding
	size_t i = 0;
	for (; i + 8 <= memorySize; i += 8)
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


// ---- flat element access (private helpers via member methods) --------------

float NDArray::loadAsFloat(size_t i) const {
	switch (type) {
		case BINARY:
			return (float)binaryGet(uint64, i);
		case UINT8:
			return (float)uint8[i];
		case F32:
			return float32[i];
		case UINT256:
			return (float)uint256[i].toDouble();
		default:
			throw std::runtime_error("NDArray::loadAsFloat: invalid type");
	}
}

uint256_t NDArray::loadAsU256(size_t i) const {
	switch (type) {
		case BINARY:
			return uint256_t((uint64_t)binaryGet(uint64, i));
		case UINT8:
			return uint256_t((uint64_t)uint8[i]);
		case F32:
			return uint256_t((double)float32[i]);
		case UINT256:
			return uint256[i];
		default:
			throw std::runtime_error("NDArray::loadAsU256: invalid type");
	}
}

void NDArray::storeFromFloat(size_t i, float v) {
	switch (type) {
		case BINARY:
			binarySet(uint64, i, v > 0.0f ? 1 : 0);
			break;
		case UINT8:
			uint8[i] = (uint8_t)v;
			break;
		case F32:
			float32[i] = v;
			break;
		case UINT256:
			uint256[i] = uint256_t((double)v);
			break;
		default:
			throw std::runtime_error("NDArray::storeFromFloat: invalid type");
	}
}

void NDArray::storeFromU256(size_t i, const uint256_t& v) {
	switch (type) {
		case BINARY:
			binarySet(uint64, i, (uint64_t)v != 0 ? 1 : 0);
			break;
		case UINT8:
			uint8[i] = (uint8_t)(uint64_t)v;
			break;
		case F32:
			float32[i] = (float)v.toDouble();
			break;
		case UINT256:
			uint256[i] = v;
			break;
		default:
			throw std::runtime_error("NDArray::storeFromU256: invalid type");
	}
}


// ---- conversion ------------------------------------------------------------

NDArray NDArray::convert(NDArrayType newType) const {
	NDArray result(shape, newType);
	const size_t n = numElements();

	if (newType == type) {
		if (memorySize)
			memcpy(result.memory, memory, memorySize);
		return result;
	}

	// Integer-only path keeps exact values (no float rounding).
	// Any involvement of F32 goes through float.
	if (isFloatingPoint(type) || isFloatingPoint(newType)) {
		for (size_t i = 0; i < n; ++i)
			result.storeFromFloat(i, loadAsFloat(i));
	} else {
		for (size_t i = 0; i < n; ++i)
			result.storeFromU256(i, loadAsU256(i));
	}
	return result;
}

void NDArray::promoteInPlace(NDArrayType newType) {
	if (newType == type)
		return;

	// convert() builds a full copy in the wider type; steal its buffer.
	NDArray converted = convert(newType);
	free(memory);
	memory = converted.memory;
	memorySize = converted.memorySize;
	type = newType;
	converted.memory = nullptr;
	converted.memorySize = 0;
}


// ---- in-place same-type kernels --------------------------------------------

void NDArray::applyBinaryInPlace(const NDArray& src, ArithOp op) {
	const size_t n = numElements();
	switch (type) {
		case F32:
			switch (op) {
				case ArithOp::Add: addBasic(float32, src.float32, n); break;
				case ArithOp::Sub: subBasic(float32, src.float32, n); break;
				case ArithOp::Mul: mulBasic(float32, src.float32, n); break;
				case ArithOp::Div: divBasic(float32, src.float32, n); break;
			}
			break;
		case UINT8:
			switch (op) {
				case ArithOp::Add: addBasic(uint8, src.uint8, n); break;
				case ArithOp::Sub: subBasic(uint8, src.uint8, n); break;
				case ArithOp::Mul: mulBasic(uint8, src.uint8, n); break;
				case ArithOp::Div: divBasic(uint8, src.uint8, n); break;
			}
			break;
		case UINT256:
			// uint256_t has +,-,*,/ but no *=; /= is const-broken — assign explicitly.
			switch (op) {
				case ArithOp::Add: addBasic(uint256, src.uint256, n); break;
				case ArithOp::Sub: subBasic(uint256, src.uint256, n); break;
				case ArithOp::Mul:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = uint256[i] * src.uint256[i];
					break;
				case ArithOp::Div:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = uint256[i] / src.uint256[i];
					break;
			}
			break;
		case BINARY:
			// Single-bit modular arithmetic (result stays in {0,1}).
			for (size_t i = 0; i < n; ++i) {
				uint8_t a = binaryGet(uint64, i);
				uint8_t b = binaryGet(src.uint64, i);
				uint8_t r = 0;
				switch (op) {
					case ArithOp::Add: r = (uint8_t)((a + b) & 1); break;
					case ArithOp::Sub: r = (uint8_t)((a - b) & 1); break;
					case ArithOp::Mul: r = (uint8_t)(a & b); break;
					case ArithOp::Div:
						if (b == 0)
							throw std::invalid_argument("NDArray: division by zero");
						r = a; // 0/1=0, 1/1=1
						break;
				}
				binarySet(uint64, i, r);
			}
			break;
		default:
			throw std::runtime_error("NDArray: invalid type in arithmetic");
	}
}

void NDArray::applyFloatScalarInPlace(float scalar, ArithOp op) {
	const size_t n = numElements();
	switch (type) {
		case F32:
			switch (op) {
				case ArithOp::Add: addBasic(float32, scalar, n); break;
				case ArithOp::Sub: subBasic(float32, scalar, n); break;
				case ArithOp::Mul: mulBasic(float32, scalar, n); break;
				case ArithOp::Div: divBasic(float32, scalar, n); break;
			}
			break;
		case UINT8: {
			uint8_t s = (uint8_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(uint8, s, n); break;
				case ArithOp::Sub: subBasic(uint8, s, n); break;
				case ArithOp::Mul: mulBasic(uint8, s, n); break;
				case ArithOp::Div: divBasic(uint8, s, n); break;
			}
			break;
		}
		case UINT256: {
			uint256_t s((double)scalar);
			switch (op) {
				case ArithOp::Add: addBasic(uint256, s, n); break;
				case ArithOp::Sub: subBasic(uint256, s, n); break;
				case ArithOp::Mul:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = uint256[i] * s;
					break;
				case ArithOp::Div:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = uint256[i] / s;
					break;
			}
			break;
		}
		case BINARY: {
			uint8_t s = scalar > 0.0f ? 1 : 0;
			for (size_t i = 0; i < n; ++i) {
				uint8_t a = binaryGet(uint64, i);
				uint8_t r = 0;
				switch (op) {
					case ArithOp::Add: r = (uint8_t)((a + s) & 1); break;
					case ArithOp::Sub: r = (uint8_t)((a - s) & 1); break;
					case ArithOp::Mul: r = (uint8_t)(a & s); break;
					case ArithOp::Div:
						if (s == 0)
							throw std::invalid_argument("NDArray: division by zero");
						r = a;
						break;
				}
				binarySet(uint64, i, r);
			}
			break;
		}
		default:
			throw std::runtime_error("NDArray: invalid type in scalar arithmetic");
	}
}

void NDArray::applyIntScalarInPlace(int scalar, ArithOp op) {
	const size_t n = numElements();
	switch (type) {
		case F32:
			applyFloatScalarInPlace((float)scalar, op);
			break;
		case UINT8: {
			uint8_t s = (uint8_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(uint8, s, n); break;
				case ArithOp::Sub: subBasic(uint8, s, n); break;
				case ArithOp::Mul: mulBasic(uint8, s, n); break;
				case ArithOp::Div: divBasic(uint8, s, n); break;
			}
			break;
		}
		case UINT256: {
			uint256_t s(scalar);
			switch (op) {
				case ArithOp::Add: addBasic(uint256, s, n); break;
				case ArithOp::Sub: subBasic(uint256, s, n); break;
				case ArithOp::Mul:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = uint256[i] * s;
					break;
				case ArithOp::Div:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = uint256[i] / s;
					break;
			}
			break;
		}
		case BINARY: {
			uint8_t s = scalar > 0 ? 1 : 0;
			for (size_t i = 0; i < n; ++i) {
				uint8_t a = binaryGet(uint64, i);
				uint8_t r = 0;
				switch (op) {
					case ArithOp::Add: r = (uint8_t)((a + s) & 1); break;
					case ArithOp::Sub: r = (uint8_t)((a - s) & 1); break;
					case ArithOp::Mul: r = (uint8_t)(a & s); break;
					case ArithOp::Div:
						if (s == 0)
							throw std::invalid_argument("NDArray: division by zero");
						r = a;
						break;
				}
				binarySet(uint64, i, r);
			}
			break;
		}
		default:
			throw std::runtime_error("NDArray: invalid type in scalar arithmetic");
	}
}


// ---- named / compound in-place ops (promote destination when needed) -------

NDArray& NDArray::binaryOpInPlace(const NDArray& other, ArithOp op) {
	requireSameShape(*this, other);
	NDArrayType resultType = promoteTypes(type, other.type);
	promoteInPlace(resultType);

	if (other.type == resultType) {
		applyBinaryInPlace(other, op);
	} else {
		NDArray tmp = other.convert(resultType);
		applyBinaryInPlace(tmp, op);
	}
	return *this;
}

NDArray& NDArray::scalarFloatOpInPlace(float other, ArithOp op) {
	NDArrayType resultType = promoteWithFloatScalar(type);
	promoteInPlace(resultType);
	applyFloatScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::scalarIntOpInPlace(int other, ArithOp op) {
	NDArrayType resultType = promoteWithIntScalar(type);
	promoteInPlace(resultType);
	if (resultType == F32)
		applyFloatScalarInPlace((float)other, op);
	else
		applyIntScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::add(NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Add);
}

NDArray& NDArray::sub(NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Sub);
}

NDArray& NDArray::mul(NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Mul);
}

NDArray& NDArray::div(NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Div);
}

NDArray& NDArray::broadcastOpInPlace(const NDArray& other, ArithOp op) {
	if (!isPrefixShape(other.shape, shape))
		throw std::invalid_argument("NDArray: other.shape must be a prefix of this->shape for broadcast");

	NDArrayType resultType = promoteTypes(type, other.type);
	promoteInPlace(resultType);

	// Point at `other` if already the result type; otherwise convert into a temporary.
	// (operator= is deleted, so we branch rather than reassign.)
	if (other.type == resultType) {
		applyBroadcastInPlace(other, op);
	} else {
		NDArray tmp = other.convert(resultType);
		applyBroadcastInPlace(tmp, op);
	}
	return *this;
}

void NDArray::applyBroadcastInPlace(const NDArray& src, ArithOp op) {
	// Precondition: src.type == this->type, src.shape is a prefix of this->shape.
	const size_t thisN = numElements();
	const size_t otherN = src.numElements();
	const size_t block = (otherN == 0) ? thisN : (thisN / otherN);

	// src indexes the outer dimensions; each src[i] is applied to a contiguous block.
	for (size_t i = 0; i < otherN; ++i) {
		for (size_t j = 0; j < block; ++j) {
			const size_t idx = i * block + j;
			switch (type) {
				case F32:
					switch (op) {
						case ArithOp::Add: float32[idx] += src.float32[i]; break;
						case ArithOp::Sub: float32[idx] -= src.float32[i]; break;
						case ArithOp::Mul: float32[idx] *= src.float32[i]; break;
						case ArithOp::Div: float32[idx] /= src.float32[i]; break;
					}
					break;
				case UINT8:
					switch (op) {
						case ArithOp::Add:
							uint8[idx] = (uint8_t)(uint8[idx] + src.uint8[i]);
							break;
						case ArithOp::Sub:
							uint8[idx] = (uint8_t)(uint8[idx] - src.uint8[i]);
							break;
						case ArithOp::Mul:
							uint8[idx] = (uint8_t)(uint8[idx] * src.uint8[i]);
							break;
						case ArithOp::Div:
							uint8[idx] = (uint8_t)(uint8[idx] / src.uint8[i]);
							break;
					}
					break;
				case UINT256:
					switch (op) {
						case ArithOp::Add:
							uint256[idx] += src.uint256[i];
							break;
						case ArithOp::Sub:
							uint256[idx] -= src.uint256[i];
							break;
						case ArithOp::Mul:
							uint256[idx] = uint256[idx] * src.uint256[i];
							break;
						case ArithOp::Div:
							uint256[idx] = uint256[idx] / src.uint256[i];
							break;
					}
					break;
				case BINARY: {
					uint8_t a = binaryGet(uint64, idx);
					uint8_t b = binaryGet(src.uint64, i);
					uint8_t r = 0;
					switch (op) {
						case ArithOp::Add: r = (uint8_t)((a + b) & 1); break;
						case ArithOp::Sub: r = (uint8_t)((a - b) & 1); break;
						case ArithOp::Mul: r = (uint8_t)(a & b); break;
						case ArithOp::Div:
							if (b == 0)
								throw std::invalid_argument("NDArray: division by zero");
							r = a;
							break;
					}
					binarySet(uint64, idx, r);
					break;
				}
				default:
					throw std::runtime_error("NDArray: invalid type in broadcast arithmetic");
			}
		}
	}
}

NDArray& NDArray::broadcastAdd(NDArray& other) {
	return broadcastOpInPlace(other, ArithOp::Add);
}

NDArray& NDArray::broadcastSub(NDArray& other) {
	return broadcastOpInPlace(other, ArithOp::Sub);
}

NDArray& NDArray::broadcastMul(NDArray& other) {
	return broadcastOpInPlace(other, ArithOp::Mul);
}

NDArray& NDArray::broadcastDiv(NDArray& other) {
	return broadcastOpInPlace(other, ArithOp::Div);
}


// ---- binary operators (promote, return new array) --------------------------

NDArray NDArray::binaryOp(const NDArray& other, ArithOp op) const {
	requireSameShape(*this, other);
	NDArrayType resultType = promoteTypes(type, other.type);
	NDArray left = convert(resultType);
	NDArray right = other.convert(resultType);
	left.applyBinaryInPlace(right, op);
	return left;
}

NDArray NDArray::scalarFloatOp(float other, ArithOp op) const {
	NDArrayType resultType = promoteWithFloatScalar(type);
	NDArray result = convert(resultType);
	result.applyFloatScalarInPlace(other, op);
	return result;
}

NDArray NDArray::scalarIntOp(int other, ArithOp op) const {
	NDArrayType resultType = promoteWithIntScalar(type);
	NDArray result = convert(resultType);
	if (resultType == F32)
		result.applyFloatScalarInPlace((float)other, op);
	else
		result.applyIntScalarInPlace(other, op);
	return result;
}


NDArray NDArray::operator+(const NDArray& other) const { return binaryOp(other, ArithOp::Add); }
NDArray NDArray::operator-(const NDArray& other) const { return binaryOp(other, ArithOp::Sub); }
NDArray NDArray::operator*(const NDArray& other) const { return binaryOp(other, ArithOp::Mul); }
NDArray NDArray::operator/(const NDArray& other) const { return binaryOp(other, ArithOp::Div); }

NDArray NDArray::operator+(float other) const { return scalarFloatOp(other, ArithOp::Add); }
NDArray NDArray::operator-(float other) const { return scalarFloatOp(other, ArithOp::Sub); }
NDArray NDArray::operator*(float other) const { return scalarFloatOp(other, ArithOp::Mul); }
NDArray NDArray::operator/(float other) const { return scalarFloatOp(other, ArithOp::Div); }

NDArray NDArray::operator+(double other) const { return scalarFloatOp((float)other, ArithOp::Add); }
NDArray NDArray::operator-(double other) const { return scalarFloatOp((float)other, ArithOp::Sub); }
NDArray NDArray::operator*(double other) const { return scalarFloatOp((float)other, ArithOp::Mul); }
NDArray NDArray::operator/(double other) const { return scalarFloatOp((float)other, ArithOp::Div); }

NDArray NDArray::operator+(int other) const { return scalarIntOp(other, ArithOp::Add); }
NDArray NDArray::operator-(int other) const { return scalarIntOp(other, ArithOp::Sub); }
NDArray NDArray::operator*(int other) const { return scalarIntOp(other, ArithOp::Mul); }
NDArray NDArray::operator/(int other) const { return scalarIntOp(other, ArithOp::Div); }


// ---- compound assignment (in place; promote destination when needed) -------

NDArray& NDArray::operator+=(const NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Add);
}
NDArray& NDArray::operator-=(const NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Sub);
}
NDArray& NDArray::operator*=(const NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Mul);
}
NDArray& NDArray::operator/=(const NDArray& other) {
	return binaryOpInPlace(other, ArithOp::Div);
}

NDArray& NDArray::operator+=(float other) {
	return scalarFloatOpInPlace(other, ArithOp::Add);
}
NDArray& NDArray::operator-=(float other) {
	return scalarFloatOpInPlace(other, ArithOp::Sub);
}
NDArray& NDArray::operator*=(float other) {
	return scalarFloatOpInPlace(other, ArithOp::Mul);
}
NDArray& NDArray::operator/=(float other) {
	return scalarFloatOpInPlace(other, ArithOp::Div);
}

NDArray& NDArray::operator+=(double other) { return *this += (float)other; }
NDArray& NDArray::operator-=(double other) { return *this -= (float)other; }
NDArray& NDArray::operator*=(double other) { return *this *= (float)other; }
NDArray& NDArray::operator/=(double other) { return *this /= (float)other; }

NDArray& NDArray::operator+=(int other) {
	return scalarIntOpInPlace(other, ArithOp::Add);
}
NDArray& NDArray::operator-=(int other) {
	return scalarIntOpInPlace(other, ArithOp::Sub);
}
NDArray& NDArray::operator*=(int other) {
	return scalarIntOpInPlace(other, ArithOp::Mul);
}
NDArray& NDArray::operator/=(int other) {
	return scalarIntOpInPlace(other, ArithOp::Div);
}



// ---- element-wise unary ----------------------------------------------------

/** Type required to apply a real-valued transcendental / root op without loss of domain. */
static NDArrayType typeForRealUnary(NDArrayType t) {
	if (t == F32)
		return F32;
	// Integers promote to F32 (only float storage type available).
	if (t == UINT256)
		throw std::invalid_argument("NDArray: real unary op not supported on UINT256 without F64/BIGINT");
	return F32;
}

/** Type for neg: needs a signed representation. */
static NDArrayType typeForNeg(NDArrayType t) {
	if (t == F32)
		return F32;
	// UINT256 has two's-complement unary minus; keep it exact.
	if (t == UINT256)
		return UINT256;
	// BINARY / UINT8 → F32
	return F32;
}

NDArray NDArray::neg() const {
	NDArrayType rt = typeForNeg(type);
	NDArray out = convert(rt);
	const size_t n = out.numElements();
	switch (out.type) {
		case F32:
			for (size_t i = 0; i < n; ++i)
				out.float32[i] = -out.float32[i];
			break;
		case UINT256:
			for (size_t i = 0; i < n; ++i)
				out.uint256[i] = -out.uint256[i];
			break;
		default:
			throw std::runtime_error("NDArray::neg: unexpected type");
	}
	return out;
}

NDArray NDArray::operator-() const {
	return neg();
}

NDArray NDArray::abs() const {
	// Unsigned / binary: identity (copy). F32: fabs. UINT256: identity as unsigned.
	if (type != F32)
		return convert(type);

	NDArray out = convert(F32);
	const size_t n = out.numElements();
	for (size_t i = 0; i < n; ++i)
		out.float32[i] = std::fabs(out.float32[i]);
	return out;
}

NDArray NDArray::sqrt() const {
	NDArray out = convert(typeForRealUnary(type));
	const size_t n = out.numElements();
	for (size_t i = 0; i < n; ++i)
		out.float32[i] = std::sqrt(out.float32[i]);
	return out;
}

NDArray NDArray::exp() const {
	NDArray out = convert(typeForRealUnary(type));
	const size_t n = out.numElements();
	for (size_t i = 0; i < n; ++i)
		out.float32[i] = std::exp(out.float32[i]);
	return out;
}

NDArray NDArray::log() const {
	NDArray out = convert(typeForRealUnary(type));
	const size_t n = out.numElements();
	for (size_t i = 0; i < n; ++i)
		out.float32[i] = std::log(out.float32[i]);
	return out;
}

NDArray NDArray::floor() const {
	if (!isFloatingPoint(type))
		return convert(type);
	NDArray out = convert(F32);
	const size_t n = out.numElements();
	for (size_t i = 0; i < n; ++i)
		out.float32[i] = std::floor(out.float32[i]);
	return out;
}

NDArray NDArray::ceil() const {
	if (!isFloatingPoint(type))
		return convert(type);
	NDArray out = convert(F32);
	const size_t n = out.numElements();
	for (size_t i = 0; i < n; ++i)
		out.float32[i] = std::ceil(out.float32[i]);
	return out;
}

NDArray NDArray::round() const {
	if (!isFloatingPoint(type))
		return convert(type);
	NDArray out = convert(F32);
	const size_t n = out.numElements();
	for (size_t i = 0; i < n; ++i)
		out.float32[i] = std::round(out.float32[i]);
	return out;
}


// ---- element-wise binary / compare / reduce (Impl has storage access) ------

enum class ElemBinOp { Min, Max, Pow, Mod };
enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };
enum class ReduceOp { Sum, Prod, Min, Max, Mean };

static float elemBinFloat(float a, float b, ElemBinOp op) {
	switch (op) {
		case ElemBinOp::Min: return std::fmin(a, b);
		case ElemBinOp::Max: return std::fmax(a, b);
		case ElemBinOp::Pow: return std::pow(a, b);
		case ElemBinOp::Mod: return std::fmod(a, b);
	}
	return 0.0f;
}

static uint8_t elemBinU8(uint8_t a, uint8_t b, ElemBinOp op) {
	switch (op) {
		case ElemBinOp::Min: return a < b ? a : b;
		case ElemBinOp::Max: return a > b ? a : b;
		case ElemBinOp::Pow: {
			uint8_t r = 1;
			for (uint8_t i = 0; i < b; ++i)
				r = (uint8_t)(r * a);
			return r;
		}
		case ElemBinOp::Mod:
			if (b == 0)
				throw std::invalid_argument("NDArray::mod: division by zero");
			return (uint8_t)(a % b);
	}
	return 0;
}

static uint256_t elemBinU256(const uint256_t& a, const uint256_t& b, ElemBinOp op) {
	switch (op) {
		case ElemBinOp::Min: return a < b ? a : b;
		case ElemBinOp::Max: return a > b ? a : b;
		case ElemBinOp::Pow: {
			uint64_t exp = (uint64_t)b;
			uint256_t r(1);
			uint256_t base = a;
			while (exp) {
				if (exp & 1ULL)
					r = r * base;
				base = base * base;
				exp >>= 1;
			}
			return r;
		}
		case ElemBinOp::Mod:
			if (b == uint256_t(0))
				throw std::invalid_argument("NDArray::mod: division by zero");
			return a % b;
	}
	return uint256_t(0);
}

static bool cmpFloat(float a, float b, CmpOp op) {
	switch (op) {
		case CmpOp::Eq: return a == b;
		case CmpOp::Ne: return a != b;
		case CmpOp::Lt: return a < b;
		case CmpOp::Le: return a <= b;
		case CmpOp::Gt: return a > b;
		case CmpOp::Ge: return a >= b;
	}
	return false;
}

static bool cmpU8(uint8_t a, uint8_t b, CmpOp op) {
	switch (op) {
		case CmpOp::Eq: return a == b;
		case CmpOp::Ne: return a != b;
		case CmpOp::Lt: return a < b;
		case CmpOp::Le: return a <= b;
		case CmpOp::Gt: return a > b;
		case CmpOp::Ge: return a >= b;
	}
	return false;
}

static bool cmpU256(const uint256_t& a, const uint256_t& b, CmpOp op) {
	switch (op) {
		case CmpOp::Eq: return a == b;
		case CmpOp::Ne: return !(a == b);
		case CmpOp::Lt: return a < b;
		case CmpOp::Le: return a < b || a == b;
		case CmpOp::Gt: return b < a;
		case CmpOp::Ge: return b < a || a == b;
	}
	return false;
}

static NDArrayType promoteForElemBin(NDArrayType a, NDArrayType b, ElemBinOp op) {
	if (op == ElemBinOp::Min || op == ElemBinOp::Max) {
		if (a == b)
			return a;
		if (isLosslessConversion(a, b))
			return b;
		if (isLosslessConversion(b, a))
			return a;
		throw std::invalid_argument("NDArray: no lossless common type for operands");
	}
	return promoteTypes(a, b);
}

static NDArrayType sumProdAccumulateType(NDArrayType t) {
	if (t == F32)
		return F32;
	return UINT256;
}

static ArrayList<int> shapeWithoutAxis(const ArrayList<int>& shape, int axis) {
	if (axis < 0 || axis >= shape.size())
		throw std::out_of_range("NDArray: reduction axis out of range");
	ArrayList<int> out;
	for (int i = 0; i < shape.size(); ++i)
		if (i != axis)
			out.add(shape.get(i));
	return out;
}

static size_t flatIndexWithAxis(const ArrayList<int>& fullShape, int axis, size_t outerInnerFlat, size_t reducedIndex) {
	const int rank = fullShape.size();
	size_t inner = 1;
	for (int d = axis + 1; d < rank; ++d)
		inner *= (size_t)fullShape.get(d);
	size_t reduced = (size_t)fullShape.get(axis);
	size_t outer = outerInnerFlat / inner;
	size_t inn = outerInnerFlat % inner;
	return outer * (reduced * inner) + reducedIndex * inner + inn;
}

static size_t axisLength(const ArrayList<int>& shape, int axis) {
	return (size_t)shape.get(axis);
}

static size_t resultElemsWithoutAxis(const ArrayList<int>& shape, int axis) {
	size_t n = 1;
	for (int i = 0; i < shape.size(); ++i)
		if (i != axis)
			n *= (size_t)shape.get(i);
	return n;
}

struct NDArray::Impl {
	static void applyElemBin(NDArray& left, const NDArray& right, ElemBinOp op) {
		const size_t n = left.numElements();
		switch (left.type) {
			case F32:
				for (size_t i = 0; i < n; ++i)
					left.float32[i] = elemBinFloat(left.float32[i], right.float32[i], op);
				break;
			case UINT8:
				for (size_t i = 0; i < n; ++i)
					left.uint8[i] = elemBinU8(left.uint8[i], right.uint8[i], op);
				break;
			case UINT256:
				for (size_t i = 0; i < n; ++i)
					left.uint256[i] = elemBinU256(left.uint256[i], right.uint256[i], op);
				break;
			case BINARY:
				for (size_t i = 0; i < n; ++i) {
					uint8_t a = binaryGet(left.uint64, i);
					uint8_t b = binaryGet(right.uint64, i);
					binarySet(left.uint64, i, elemBinU8(a, b, op) & 1);
				}
				break;
			default:
				throw std::runtime_error("NDArray: invalid type in element-wise binary op");
		}
	}

	static NDArray elemBinArrays(const NDArray& a, const NDArray& b, ElemBinOp op) {
		requireSameShape(a, b);
		NDArrayType rt = promoteForElemBin(a.type, b.type, op);
		NDArray left = a.convert(rt);
		NDArray right = b.convert(rt);
		applyElemBin(left, right, op);
		return left;
	}

	static NDArray elemBinFloatScalar(const NDArray& a, float s, ElemBinOp op) {
		NDArrayType rt = promoteWithFloatScalar(a.type);
		if ((op == ElemBinOp::Min || op == ElemBinOp::Max) && isLosslessConversion(a.type, F32))
			rt = F32;
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (left.type == F32) {
			for (size_t i = 0; i < n; ++i)
				left.float32[i] = elemBinFloat(left.float32[i], s, op);
		} else if (left.type == UINT8) {
			uint8_t sb = (uint8_t)s;
			for (size_t i = 0; i < n; ++i)
				left.uint8[i] = elemBinU8(left.uint8[i], sb, op);
		} else if (left.type == UINT256) {
			uint256_t sb((double)s);
			for (size_t i = 0; i < n; ++i)
				left.uint256[i] = elemBinU256(left.uint256[i], sb, op);
		} else {
			throw std::runtime_error("NDArray: invalid type in scalar element-wise op");
		}
		return left;
	}

	static NDArray elemBinIntScalar(const NDArray& a, int s, ElemBinOp op) {
		NDArrayType rt = promoteWithIntScalar(a.type);
		if (op == ElemBinOp::Min || op == ElemBinOp::Max) {
			if (a.type == UINT8 && s >= 0 && s <= 255)
				rt = UINT8;
			else if (a.type == UINT256 && s >= 0)
				rt = UINT256;
			else if (a.type == BINARY && (s == 0 || s == 1))
				rt = BINARY;
			else if (a.type == F32)
				rt = F32;
			else
				rt = promoteWithIntScalar(a.type);
		}
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (left.type == F32) {
			float sf = (float)s;
			for (size_t i = 0; i < n; ++i)
				left.float32[i] = elemBinFloat(left.float32[i], sf, op);
		} else if (left.type == UINT8) {
			uint8_t sb = (uint8_t)s;
			for (size_t i = 0; i < n; ++i)
				left.uint8[i] = elemBinU8(left.uint8[i], sb, op);
		} else if (left.type == UINT256) {
			uint256_t sb(s);
			for (size_t i = 0; i < n; ++i)
				left.uint256[i] = elemBinU256(left.uint256[i], sb, op);
		} else if (left.type == BINARY) {
			uint8_t sb = s > 0 ? 1 : 0;
			for (size_t i = 0; i < n; ++i) {
				uint8_t av = binaryGet(left.uint64, i);
				binarySet(left.uint64, i, elemBinU8(av, sb, op) & 1);
			}
		} else {
			throw std::runtime_error("NDArray: invalid type in scalar element-wise op");
		}
		return left;
	}

	static NDArray compareArrays(const NDArray& a, const NDArray& b, CmpOp op) {
		requireSameShape(a, b);
		NDArrayType ct;
		if (a.type == b.type)
			ct = a.type;
		else if (isLosslessConversion(a.type, b.type))
			ct = b.type;
		else if (isLosslessConversion(b.type, a.type))
			ct = a.type;
		else
			throw std::invalid_argument("NDArray: no lossless common type for comparison");

		NDArray left = a.convert(ct);
		NDArray right = b.convert(ct);
		NDArray out(a.shape, BINARY);
		const size_t n = left.numElements();

		for (size_t i = 0; i < n; ++i) {
			bool r = false;
			switch (ct) {
				case F32:
					r = cmpFloat(left.float32[i], right.float32[i], op);
					break;
				case UINT8:
					r = cmpU8(left.uint8[i], right.uint8[i], op);
					break;
				case UINT256:
					r = cmpU256(left.uint256[i], right.uint256[i], op);
					break;
				case BINARY:
					r = cmpU8(binaryGet(left.uint64, i), binaryGet(right.uint64, i), op);
					break;
				default:
					throw std::runtime_error("NDArray: invalid type in comparison");
			}
			binarySet(out.uint64, i, r ? 1 : 0);
		}
		return out;
	}

	static NDArray compareFloatScalar(const NDArray& a, float s, CmpOp op) {
		NDArrayType ct = F32;
		if (!isLosslessConversion(a.type, F32) && a.type != F32)
			throw std::invalid_argument("NDArray: cannot compare this type with float scalar");
		if (a.type == F32 || isLosslessConversion(a.type, F32))
			ct = F32;

		NDArray left = a.convert(ct);
		NDArray out(a.shape, BINARY);
		const size_t n = left.numElements();
		for (size_t i = 0; i < n; ++i) {
			bool r = false;
			if (ct == F32)
				r = cmpFloat(left.float32[i], s, op);
			else
				throw std::runtime_error("NDArray: unexpected compare scalar type");
			binarySet(out.uint64, i, r ? 1 : 0);
		}
		return out;
	}

	static NDArray where(const NDArray& condition, const NDArray& x, const NDArray& y) {
		requireSameShape(condition, x);
		requireSameShape(condition, y);

		NDArray mask(condition.shape, BINARY);
		const size_t n = condition.numElements();
		for (size_t i = 0; i < n; ++i) {
			bool t = false;
			switch (condition.type) {
				case F32: t = condition.float32[i] != 0.0f; break;
				case UINT8: t = condition.uint8[i] != 0; break;
				case UINT256: t = !(condition.uint256[i] == uint256_t(0)); break;
				case BINARY: t = binaryGet(condition.uint64, i) != 0; break;
				default: break;
			}
			binarySet(mask.uint64, i, t ? 1 : 0);
		}

		NDArrayType rt;
		if (x.type == y.type)
			rt = x.type;
		else
			rt = promoteTypes(x.type, y.type);

		NDArray xv = x.convert(rt);
		NDArray yv = y.convert(rt);
		NDArray out(condition.shape, rt);

		for (size_t i = 0; i < n; ++i) {
			bool takeX = binaryGet(mask.uint64, i) != 0;
			switch (rt) {
				case F32:
					out.float32[i] = takeX ? xv.float32[i] : yv.float32[i];
					break;
				case UINT8:
					out.uint8[i] = takeX ? xv.uint8[i] : yv.uint8[i];
					break;
				case UINT256:
					out.uint256[i] = takeX ? xv.uint256[i] : yv.uint256[i];
					break;
				case BINARY:
					binarySet(out.uint64, i, takeX ? binaryGet(xv.uint64, i) : binaryGet(yv.uint64, i));
					break;
				default:
					throw std::runtime_error("NDArray::where: invalid type");
			}
		}
		return out;
	}

	static NDArray reduceAll(const NDArray& a, ReduceOp op) {
		const size_t n = a.numElements();
		if (n == 0)
			throw std::invalid_argument("NDArray: cannot reduce empty array");

		if (op == ReduceOp::Mean) {
			NDArray s = reduceAll(a, ReduceOp::Sum);
			NDArray out({}, F32);
			if (s.type == F32)
				out.float32[0] = s.float32[0] / (float)n;
			else
				out.float32[0] = (float)(s.uint256[0].toDouble() / (double)n);
			return out;
		}

		if (op == ReduceOp::Min || op == ReduceOp::Max) {
			NDArray out({}, a.type);
			switch (a.type) {
				case F32: {
					float acc = a.float32[0];
					for (size_t i = 1; i < n; ++i)
						acc = (op == ReduceOp::Min) ? std::fmin(acc, a.float32[i])
						                            : std::fmax(acc, a.float32[i]);
					out.float32[0] = acc;
					break;
				}
				case UINT8: {
					uint8_t acc = a.uint8[0];
					for (size_t i = 1; i < n; ++i)
						acc = (op == ReduceOp::Min)
							? (a.uint8[i] < acc ? a.uint8[i] : acc)
							: (a.uint8[i] > acc ? a.uint8[i] : acc);
					out.uint8[0] = acc;
					break;
				}
				case UINT256: {
					uint256_t acc = a.uint256[0];
					for (size_t i = 1; i < n; ++i) {
						if (op == ReduceOp::Min) {
							if (a.uint256[i] < acc) acc = a.uint256[i];
						} else {
							if (acc < a.uint256[i]) acc = a.uint256[i];
						}
					}
					out.uint256[0] = acc;
					break;
				}
				case BINARY: {
					uint8_t acc = binaryGet(a.uint64, 0);
					for (size_t i = 1; i < n; ++i) {
						uint8_t v = binaryGet(a.uint64, i);
						acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
					}
					binarySet(out.uint64, 0, acc);
					break;
				}
				default:
					throw std::runtime_error("NDArray: invalid type in min/max reduce");
			}
			return out;
		}

		NDArrayType rt = sumProdAccumulateType(a.type);
		NDArray out({}, rt);
		if (rt == F32) {
			float acc = (op == ReduceOp::Sum) ? 0.0f : 1.0f;
			for (size_t i = 0; i < n; ++i) {
				float v = a.loadAsFloat(i);
				acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
			}
			out.float32[0] = acc;
		} else {
			uint256_t acc = (op == ReduceOp::Sum) ? uint256_t(0) : uint256_t(1);
			for (size_t i = 0; i < n; ++i) {
				uint256_t v = a.loadAsU256(i);
				acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
			}
			out.uint256[0] = acc;
		}
		return out;
	}

	static NDArray reduceAxis(const NDArray& a, int axis, ReduceOp op) {
		if (a.shape.size() == 0)
			throw std::invalid_argument("NDArray: cannot reduce a scalar along an axis");
		if (axis < 0 || axis >= a.shape.size())
			throw std::out_of_range("NDArray: reduction axis out of range");

		const size_t reduced = axisLength(a.shape, axis);
		if (reduced == 0)
			throw std::invalid_argument("NDArray: cannot reduce along a zero-length axis");

		const size_t outN = resultElemsWithoutAxis(a.shape, axis);
		ArrayList<int> outShape = shapeWithoutAxis(a.shape, axis);

		if (op == ReduceOp::Mean) {
			NDArray s = reduceAxis(a, axis, ReduceOp::Sum);
			if (s.type == F32) {
				for (size_t i = 0; i < s.numElements(); ++i)
					s.float32[i] /= (float)reduced;
				return s;
			}
			NDArray out(outShape, F32);
			for (size_t i = 0; i < outN; ++i)
				out.float32[i] = (float)(s.uint256[i].toDouble() / (double)reduced);
			return out;
		}

		if (op == ReduceOp::Min || op == ReduceOp::Max) {
			NDArray out(outShape, a.type);
			for (size_t oi = 0; oi < outN; ++oi) {
				size_t first = flatIndexWithAxis(a.shape, axis, oi, 0);
				switch (a.type) {
					case F32: {
						float acc = a.float32[first];
						for (size_t r = 1; r < reduced; ++r) {
							float v = a.float32[flatIndexWithAxis(a.shape, axis, oi, r)];
							acc = (op == ReduceOp::Min) ? std::fmin(acc, v) : std::fmax(acc, v);
						}
						out.float32[oi] = acc;
						break;
					}
					case UINT8: {
						uint8_t acc = a.uint8[first];
						for (size_t r = 1; r < reduced; ++r) {
							uint8_t v = a.uint8[flatIndexWithAxis(a.shape, axis, oi, r)];
							acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
						}
						out.uint8[oi] = acc;
						break;
					}
					case UINT256: {
						uint256_t acc = a.uint256[first];
						for (size_t r = 1; r < reduced; ++r) {
							const uint256_t& v = a.uint256[flatIndexWithAxis(a.shape, axis, oi, r)];
							if (op == ReduceOp::Min) {
								if (v < acc) acc = v;
							} else {
								if (acc < v) acc = v;
							}
						}
						out.uint256[oi] = acc;
						break;
					}
					case BINARY: {
						uint8_t acc = binaryGet(a.uint64, first);
						for (size_t r = 1; r < reduced; ++r) {
							uint8_t v = binaryGet(a.uint64, flatIndexWithAxis(a.shape, axis, oi, r));
							acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
						}
						binarySet(out.uint64, oi, acc);
						break;
					}
					default:
						throw std::runtime_error("NDArray: invalid type in axis min/max");
				}
			}
			return out;
		}

		NDArrayType rt = sumProdAccumulateType(a.type);
		NDArray out(outShape, rt);
		for (size_t oi = 0; oi < outN; ++oi) {
			if (rt == F32) {
				float acc = (op == ReduceOp::Sum) ? 0.0f : 1.0f;
				for (size_t r = 0; r < reduced; ++r) {
					size_t idx = flatIndexWithAxis(a.shape, axis, oi, r);
					float v = a.loadAsFloat(idx);
					acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
				}
				out.float32[oi] = acc;
			} else {
				uint256_t acc = (op == ReduceOp::Sum) ? uint256_t(0) : uint256_t(1);
				for (size_t r = 0; r < reduced; ++r) {
					size_t idx = flatIndexWithAxis(a.shape, axis, oi, r);
					uint256_t v = a.loadAsU256(idx);
					acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
				}
				out.uint256[oi] = acc;
			}
		}
		return out;
	}
};

NDArray NDArray::minimum(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Min); }
NDArray NDArray::maximum(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Max); }
NDArray NDArray::pow(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Pow); }
NDArray NDArray::mod(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Mod); }

NDArray NDArray::minimum(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Min); }
NDArray NDArray::maximum(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Max); }
NDArray NDArray::pow(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Pow); }
NDArray NDArray::mod(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Mod); }

NDArray NDArray::minimum(int other) const { return Impl::elemBinIntScalar(*this, other, ElemBinOp::Min); }
NDArray NDArray::maximum(int other) const { return Impl::elemBinIntScalar(*this, other, ElemBinOp::Max); }
NDArray NDArray::pow(int other) const { return Impl::elemBinIntScalar(*this, other, ElemBinOp::Pow); }
NDArray NDArray::mod(int other) const { return Impl::elemBinIntScalar(*this, other, ElemBinOp::Mod); }

NDArray NDArray::equal(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Eq); }
NDArray NDArray::notEqual(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Ne); }
NDArray NDArray::less(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Lt); }
NDArray NDArray::lessEqual(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Le); }
NDArray NDArray::greater(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Gt); }
NDArray NDArray::greaterEqual(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Ge); }

NDArray NDArray::equal(float other) const { return Impl::compareFloatScalar(*this, other, CmpOp::Eq); }
NDArray NDArray::notEqual(float other) const { return Impl::compareFloatScalar(*this, other, CmpOp::Ne); }
NDArray NDArray::less(float other) const { return Impl::compareFloatScalar(*this, other, CmpOp::Lt); }
NDArray NDArray::lessEqual(float other) const { return Impl::compareFloatScalar(*this, other, CmpOp::Le); }
NDArray NDArray::greater(float other) const { return Impl::compareFloatScalar(*this, other, CmpOp::Gt); }
NDArray NDArray::greaterEqual(float other) const { return Impl::compareFloatScalar(*this, other, CmpOp::Ge); }

NDArray NDArray::where(const NDArray& condition, const NDArray& x, const NDArray& y) {
	return Impl::where(condition, x, y);
}

bool NDArray::any() const {
	const size_t n = numElements();
	for (size_t i = 0; i < n; ++i) {
		switch (type) {
			case BINARY:
				if (binaryGet(uint64, i)) return true;
				break;
			case UINT8:
				if (uint8[i]) return true;
				break;
			case F32:
				if (float32[i] != 0.0f) return true;
				break;
			case UINT256:
				if (!(uint256[i] == uint256_t(0))) return true;
				break;
			default:
				throw std::runtime_error("NDArray::any: invalid type");
		}
	}
	return false;
}

bool NDArray::all() const {
	const size_t n = numElements();
	if (n == 0)
		return true;
	for (size_t i = 0; i < n; ++i) {
		switch (type) {
			case BINARY:
				if (!binaryGet(uint64, i)) return false;
				break;
			case UINT8:
				if (!uint8[i]) return false;
				break;
			case F32:
				if (float32[i] == 0.0f) return false;
				break;
			case UINT256:
				if (uint256[i] == uint256_t(0)) return false;
				break;
			default:
				throw std::runtime_error("NDArray::all: invalid type");
		}
	}
	return true;
}

NDArray NDArray::sum() const { return Impl::reduceAll(*this, ReduceOp::Sum); }
NDArray NDArray::sum(int axis) const { return Impl::reduceAxis(*this, axis, ReduceOp::Sum); }
NDArray NDArray::mean() const { return Impl::reduceAll(*this, ReduceOp::Mean); }
NDArray NDArray::mean(int axis) const { return Impl::reduceAxis(*this, axis, ReduceOp::Mean); }
NDArray NDArray::min() const { return Impl::reduceAll(*this, ReduceOp::Min); }
NDArray NDArray::min(int axis) const { return Impl::reduceAxis(*this, axis, ReduceOp::Min); }
NDArray NDArray::max() const { return Impl::reduceAll(*this, ReduceOp::Max); }
NDArray NDArray::max(int axis) const { return Impl::reduceAxis(*this, axis, ReduceOp::Max); }
NDArray NDArray::prod() const { return Impl::reduceAll(*this, ReduceOp::Prod); }
NDArray NDArray::prod(int axis) const { return Impl::reduceAxis(*this, axis, ReduceOp::Prod); }

// Touch type-rule helpers so -Werror does not drop them before wider use.
namespace {
struct TypeRuleAnchor {
	TypeRuleAnchor() { retainTypeRuleHelpers(); }
};
static TypeRuleAnchor typeRuleAnchor;
}
