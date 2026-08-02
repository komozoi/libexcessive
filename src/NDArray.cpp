
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, except software not created by me.
// If you find this software, please notify me immediately: komozoi@protonmail.com
//
//

#include "../include/NDArray.h"

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

NDArray& NDArray::broadcastAdd(NDArray& other) {
	if (type != other.type)
		throw std::invalid_argument("NDArray::broadcastAdd: type mismatch");
	if (!isPrefixShape(other.shape, shape))
		throw std::invalid_argument("NDArray::broadcastAdd: other.shape must be a prefix of this->shape");

	const size_t thisN = numElements();
	const size_t otherN = other.numElements();
	const size_t block = (otherN == 0) ? thisN : (thisN / otherN);

	// other indexes the outer dimensions; each other[i] is added to a contiguous block.
	for (size_t i = 0; i < otherN; ++i) {
		for (size_t j = 0; j < block; ++j) {
			const size_t idx = i * block + j;
			switch (type) {
				case F32:
					float32[idx] += other.float32[i];
					break;
				case UINT8:
					uint8[idx] = (uint8_t)(uint8[idx] + other.uint8[i]);
					break;
				case UINT256:
					uint256[idx] += other.uint256[i];
					break;
				case BINARY: {
					uint8_t a = binaryGet(uint64, idx);
					uint8_t b = binaryGet(other.uint64, i);
					binarySet(uint64, idx, (uint8_t)((a + b) & 1));
					break;
				}
				default:
					throw std::runtime_error("NDArray::broadcastAdd: invalid type");
			}
		}
	}
	return *this;
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

// Touch type-rule helpers so -Werror does not drop them before wider use.
namespace {
struct TypeRuleAnchor {
	TypeRuleAnchor() { retainTypeRuleHelpers(); }
};
static TypeRuleAnchor typeRuleAnchor;
}
