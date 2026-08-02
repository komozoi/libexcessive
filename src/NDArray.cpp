
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
**
** Unary *methods* mutate in place.  Operators return new arrays (copy).
*/

static bool isFloatingPoint(NDArrayType type) {
	return type == F32 || type == F64;
}

static bool isSigned(NDArrayType type) {
	return type == F32 || type == F64 || type == INT32 || type == INT64;
}

static bool isSignedInteger(NDArrayType type) {
	return type == INT32 || type == INT64;
}

static int maxPowerOfTypeIntPrecision(NDArrayType type) {
	switch (type) {
		case BINARY:  return 0;
		case UINT8:   return 7;
		case INT32:   return 31;
		case INT64:   return 63;
		case F32:     return 24;
		case F64:     return 53;
		case UINT256: return 255;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

static int maxPowerOfType(NDArrayType type) {
	switch (type) {
		case BINARY:  return 0;
		case UINT8:   return 7;
		case INT32:   return 31;
		case INT64:   return 63;
		case F32:     return 128;
		case F64:     return 1024;
		case UINT256: return 255;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

static void retainTypeRuleHelpers() {
	(void)&maxPowerOfType;
}

static bool isLosslessConversion(NDArrayType from, NDArrayType to) {
	if (from == to)
		return true;
	// Float → integer always drops fractional information.
	if (isFloatingPoint(from) && !isFloatingPoint(to))
		return false;
	if (maxPowerOfTypeIntPrecision(to) < maxPowerOfTypeIntPrecision(from))
		return false;
	if (isSigned(from) && !isSigned(to))
		return false;
	return true;
}

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

	// If either operand is floating, prefer a float common type so fractions
	// are not forced into integers (e.g. INT32 + F32 → F64, not INT64).
	if (isFloatingPoint(a) || isFloatingPoint(b)) {
		static const NDArrayType floatCandidates[] = { F32, F64 };
		for (NDArrayType c : floatCandidates) {
			if (isLosslessConversion(a, c) && isLosslessConversion(b, c))
				return c;
		}
	}

	// Neither is a supertype of the other — try common wider types.
	static const NDArrayType candidates[] = {
		INT32, INT64, F32, F64, UINT256
	};
	for (NDArrayType c : candidates) {
		if (isLosslessConversion(a, c) && isLosslessConversion(b, c))
			return c;
	}
	throw std::invalid_argument("NDArray: no lossless common type for operands");
}

static NDArrayType promoteWithFloatScalar(NDArrayType arrayType) {
	return promoteTypes(arrayType, F32);
}

static NDArrayType promoteWithDoubleScalar(NDArrayType arrayType) {
	return promoteTypes(arrayType, F64);
}

static NDArrayType promoteWithIntScalar(NDArrayType arrayType) {
	if (arrayType == UINT256)
		return UINT256;
	return promoteTypes(arrayType, INT32);
}

/** Real unary ops (log, sin, …): stay in current float width, else promote to F64. */
static NDArrayType typeForRealUnary(NDArrayType t) {
	if (t == F32)
		return F32;
	if (t == F64)
		return F64;
	if (t == UINT256)
		throw std::invalid_argument("NDArray: real unary op not supported on UINT256 without a wider float");
	return F64;
}

static NDArrayType typeForNeg(NDArrayType t) {
	if (t == F32 || t == F64 || t == INT32 || t == INT64)
		return t;
	if (t == UINT256)
		return UINT256; // two's complement
	// BINARY / UINT8 → INT32 (exact for those ranges)
	return INT32;
}

static NDArrayType sumProdAccumulateType(NDArrayType t) {
	switch (t) {
		case F32: return F32;
		case F64: return F64;
		case INT32: return INT64;
		case INT64: return INT64;
		case BINARY:
		case UINT8:
		case UINT256:
			return UINT256;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

static NDArrayType meanResultType(NDArrayType t) {
	if (t == F32)
		return F32;
	return F64;
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
NDArray::NDArray(ArrayList<double> vector) : shape({vector.size()}), type(F64) {
	initialize();
	memcpy(float64, vector.getMemory(), memorySize);
}
NDArray::NDArray(ArrayList<uint8_t> vector) : shape({vector.size()}), type(UINT8) {
	initialize();
	memcpy(uint8, vector.getMemory(), memorySize);
}
NDArray::NDArray(ArrayList<int32_t> vector) : shape({vector.size()}), type(INT32) {
	initialize();
	memcpy(int32, vector.getMemory(), memorySize);
}
NDArray::NDArray(ArrayList<int64_t> vector) : shape({vector.size()}), type(INT64) {
	initialize();
	memcpy(int64, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int>& shape, ArrayList<float> vector) : shape(shape), type(F32) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	memcpy(float32, vector.getMemory(), memorySize);
}
NDArray::NDArray(const ArrayList<int>& shape, ArrayList<double> vector) : shape(shape), type(F64) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	memcpy(float64, vector.getMemory(), memorySize);
}
NDArray::NDArray(const ArrayList<int>& shape, ArrayList<uint8_t> vector) : shape(shape), type(UINT8) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	memcpy(uint8, vector.getMemory(), memorySize);
}
NDArray::NDArray(const ArrayList<int>& shape, ArrayList<int32_t> vector) : shape(shape), type(INT32) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	memcpy(int32, vector.getMemory(), memorySize);
}
NDArray::NDArray(const ArrayList<int>& shape, ArrayList<int64_t> vector) : shape(shape), type(INT64) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	memcpy(int64, vector.getMemory(), memorySize);
}

NDArray::NDArray(const NDArray& other) : shape(other.shape), type(other.type), memorySize(other.memorySize) {
	memory = malloc(memorySize ? memorySize : 1);
	if (memorySize)
		memcpy(memory, other.memory, memorySize);
}

NDArray::NDArray(NDArray&& other) noexcept
	: shape(other.shape), type(other.type), memorySize(other.memorySize) {
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
			memorySize = ((totalElements + 63) / 64) * 8;
			break;
		case UINT8:
			memorySize = totalElements;
			break;
		case INT32:
		case F32:
			memorySize = totalElements * 4;
			break;
		case INT64:
		case F64:
			memorySize = totalElements * 8;
			break;
		case UINT256:
			memorySize = totalElements * 32;
			break;
	}

	memory = malloc(memorySize ? memorySize : 1);
	if (memorySize)
		bzero(memory, memorySize);
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
	size_t i = 0;
	for (; i + 8 <= memorySize; i += 8)
		if (uint64[i / 8] != other.uint64[i / 8])
			return false;
	for (; i < memorySize; ++i)
		if (uint8[i] != other.uint8[i])
			return false;
	return true;
}

NDArray::~NDArray() {
	free(memory);
}

size_t NDArray::computeOffset(const ArrayList<int>& indices) const {
	size_t offset = 0;
	size_t stride = 1;
	for (int d = shape.size() - 1; d >= 0; --d) {
		int idx = indices.get(d);
		if (idx < 0 || idx >= shape.get(d))
			throw std::out_of_range("index out of bounds");
		offset += idx * stride;
		stride *= shape.get(d);
	}
	return offset;
}

void NDArray::stealFrom(NDArray& result) {
	free(memory);
	memory = result.memory;
	memorySize = result.memorySize;
	type = result.type;
	result.memory = nullptr;
	result.memorySize = 0;
}


// ---- load / store ----------------------------------------------------------

float NDArray::loadAsFloat(size_t i) const {
	return (float)loadAsDouble(i);
}

double NDArray::loadAsDouble(size_t i) const {
	switch (type) {
		case BINARY:  return (double)binaryGet(uint64, i);
		case UINT8:   return (double)uint8[i];
		case INT32:   return (double)int32[i];
		case INT64:   return (double)int64[i];
		case F32:     return (double)float32[i];
		case F64:     return float64[i];
		case UINT256: return uint256[i].toDouble();
		default: throw std::runtime_error("loadAsDouble: invalid type");
	}
}

int64_t NDArray::loadAsI64(size_t i) const {
	switch (type) {
		case BINARY:  return (int64_t)binaryGet(uint64, i);
		case UINT8:   return (int64_t)uint8[i];
		case INT32:   return (int64_t)int32[i];
		case INT64:   return int64[i];
		case F32:     return (int64_t)float32[i];
		case F64:     return (int64_t)float64[i];
		case UINT256: return (int64_t)(uint64_t)uint256[i];
		default: throw std::runtime_error("loadAsI64: invalid type");
	}
}

uint256_t NDArray::loadAsU256(size_t i) const {
	switch (type) {
		case BINARY:  return uint256_t((uint64_t)binaryGet(uint64, i));
		case UINT8:   return uint256_t((uint64_t)uint8[i]);
		case INT32:   return uint256_t((int)int32[i]);
		case INT64:   return uint256_t((uint64_t)int64[i]); // truncates sign bit interpretation
		case F32:     return uint256_t((double)float32[i]);
		case F64:     return uint256_t(float64[i]);
		case UINT256: return uint256[i];
		default: throw std::runtime_error("loadAsU256: invalid type");
	}
}

void NDArray::storeFromFloat(size_t i, float v) { storeFromDouble(i, (double)v); }

void NDArray::storeFromDouble(size_t i, double v) {
	switch (type) {
		case BINARY:  binarySet(uint64, i, v > 0.0 ? 1 : 0); break;
		case UINT8:   uint8[i] = (uint8_t)v; break;
		case INT32:   int32[i] = (int32_t)v; break;
		case INT64:   int64[i] = (int64_t)v; break;
		case F32:     float32[i] = (float)v; break;
		case F64:     float64[i] = v; break;
		case UINT256: uint256[i] = uint256_t(v); break;
		default: throw std::runtime_error("storeFromDouble: invalid type");
	}
}

void NDArray::storeFromI64(size_t i, int64_t v) {
	switch (type) {
		case BINARY:  binarySet(uint64, i, v > 0 ? 1 : 0); break;
		case UINT8:   uint8[i] = (uint8_t)v; break;
		case INT32:   int32[i] = (int32_t)v; break;
		case INT64:   int64[i] = v; break;
		case F32:     float32[i] = (float)v; break;
		case F64:     float64[i] = (double)v; break;
		case UINT256:
			if (v < 0)
				uint256[i] = uint256_t((int)v); // two's-complement via int ctor when in range
			else
				uint256[i] = uint256_t((uint64_t)v);
			break;
		default: throw std::runtime_error("storeFromI64: invalid type");
	}
}

void NDArray::storeFromU256(size_t i, const uint256_t& v) {
	switch (type) {
		case BINARY:  binarySet(uint64, i, (uint64_t)v != 0 ? 1 : 0); break;
		case UINT8:   uint8[i] = (uint8_t)(uint64_t)v; break;
		case INT32:   int32[i] = (int32_t)(uint64_t)v; break;
		case INT64:   int64[i] = (int64_t)(uint64_t)v; break;
		case F32:     float32[i] = (float)v.toDouble(); break;
		case F64:     float64[i] = v.toDouble(); break;
		case UINT256: uint256[i] = v; break;
		default: throw std::runtime_error("storeFromU256: invalid type");
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

	if (isFloatingPoint(type) || isFloatingPoint(newType)) {
		for (size_t i = 0; i < n; ++i)
			result.storeFromDouble(i, loadAsDouble(i));
	} else if (isSignedInteger(type) || isSignedInteger(newType)) {
		for (size_t i = 0; i < n; ++i)
			result.storeFromI64(i, loadAsI64(i));
	} else {
		for (size_t i = 0; i < n; ++i)
			result.storeFromU256(i, loadAsU256(i));
	}
	return result;
}

void NDArray::promoteInPlace(NDArrayType newType) {
	if (newType == type)
		return;
	NDArray converted = convert(newType);
	stealFrom(converted);
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
		case F64:
			switch (op) {
				case ArithOp::Add: addBasic(float64, src.float64, n); break;
				case ArithOp::Sub: subBasic(float64, src.float64, n); break;
				case ArithOp::Mul: mulBasic(float64, src.float64, n); break;
				case ArithOp::Div: divBasic(float64, src.float64, n); break;
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
		case INT32:
			switch (op) {
				case ArithOp::Add: addBasic(int32, src.int32, n); break;
				case ArithOp::Sub: subBasic(int32, src.int32, n); break;
				case ArithOp::Mul: mulBasic(int32, src.int32, n); break;
				case ArithOp::Div: divBasic(int32, src.int32, n); break;
			}
			break;
		case INT64:
			switch (op) {
				case ArithOp::Add: addBasic(int64, src.int64, n); break;
				case ArithOp::Sub: subBasic(int64, src.int64, n); break;
				case ArithOp::Mul: mulBasic(int64, src.int64, n); break;
				case ArithOp::Div: divBasic(int64, src.int64, n); break;
			}
			break;
		case UINT256:
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
						r = a;
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
	applyDoubleScalarInPlace((double)scalar, op);
}

void NDArray::applyDoubleScalarInPlace(double scalar, ArithOp op) {
	const size_t n = numElements();
	switch (type) {
		case F32: {
			float s = (float)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(float32, s, n); break;
				case ArithOp::Sub: subBasic(float32, s, n); break;
				case ArithOp::Mul: mulBasic(float32, s, n); break;
				case ArithOp::Div: divBasic(float32, s, n); break;
			}
			break;
		}
		case F64:
			switch (op) {
				case ArithOp::Add: addBasic(float64, scalar, n); break;
				case ArithOp::Sub: subBasic(float64, scalar, n); break;
				case ArithOp::Mul: mulBasic(float64, scalar, n); break;
				case ArithOp::Div: divBasic(float64, scalar, n); break;
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
		case INT32: {
			int32_t s = (int32_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(int32, s, n); break;
				case ArithOp::Sub: subBasic(int32, s, n); break;
				case ArithOp::Mul: mulBasic(int32, s, n); break;
				case ArithOp::Div: divBasic(int32, s, n); break;
			}
			break;
		}
		case INT64: {
			int64_t s = (int64_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(int64, s, n); break;
				case ArithOp::Sub: subBasic(int64, s, n); break;
				case ArithOp::Mul: mulBasic(int64, s, n); break;
				case ArithOp::Div: divBasic(int64, s, n); break;
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
			uint8_t s = scalar > 0.0 ? 1 : 0;
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
		case F64:
			applyDoubleScalarInPlace((double)scalar, op);
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
		case INT32: {
			int32_t s = (int32_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(int32, s, n); break;
				case ArithOp::Sub: subBasic(int32, s, n); break;
				case ArithOp::Mul: mulBasic(int32, s, n); break;
				case ArithOp::Div: divBasic(int32, s, n); break;
			}
			break;
		}
		case INT64: {
			int64_t s = (int64_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(int64, s, n); break;
				case ArithOp::Sub: subBasic(int64, s, n); break;
				case ArithOp::Mul: mulBasic(int64, s, n); break;
				case ArithOp::Div: divBasic(int64, s, n); break;
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


// ---- named / compound in-place ---------------------------------------------

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
	promoteInPlace(promoteWithFloatScalar(type));
	applyFloatScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::scalarDoubleOpInPlace(double other, ArithOp op) {
	promoteInPlace(promoteWithDoubleScalar(type));
	applyDoubleScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::scalarIntOpInPlace(int other, ArithOp op) {
	promoteInPlace(promoteWithIntScalar(type));
	if (type == F32 || type == F64)
		applyDoubleScalarInPlace((double)other, op);
	else
		applyIntScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::add(NDArray& other) { return binaryOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::sub(NDArray& other) { return binaryOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::mul(NDArray& other) { return binaryOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::div(NDArray& other) { return binaryOpInPlace(other, ArithOp::Div); }

void NDArray::applyBroadcastInPlace(const NDArray& src, ArithOp op) {
	const size_t thisN = numElements();
	const size_t otherN = src.numElements();
	const size_t block = (otherN == 0) ? thisN : (thisN / otherN);

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
				case F64:
					switch (op) {
						case ArithOp::Add: float64[idx] += src.float64[i]; break;
						case ArithOp::Sub: float64[idx] -= src.float64[i]; break;
						case ArithOp::Mul: float64[idx] *= src.float64[i]; break;
						case ArithOp::Div: float64[idx] /= src.float64[i]; break;
					}
					break;
				case UINT8:
					switch (op) {
						case ArithOp::Add: uint8[idx] = (uint8_t)(uint8[idx] + src.uint8[i]); break;
						case ArithOp::Sub: uint8[idx] = (uint8_t)(uint8[idx] - src.uint8[i]); break;
						case ArithOp::Mul: uint8[idx] = (uint8_t)(uint8[idx] * src.uint8[i]); break;
						case ArithOp::Div: uint8[idx] = (uint8_t)(uint8[idx] / src.uint8[i]); break;
					}
					break;
				case INT32:
					switch (op) {
						case ArithOp::Add: int32[idx] += src.int32[i]; break;
						case ArithOp::Sub: int32[idx] -= src.int32[i]; break;
						case ArithOp::Mul: int32[idx] *= src.int32[i]; break;
						case ArithOp::Div: int32[idx] /= src.int32[i]; break;
					}
					break;
				case INT64:
					switch (op) {
						case ArithOp::Add: int64[idx] += src.int64[i]; break;
						case ArithOp::Sub: int64[idx] -= src.int64[i]; break;
						case ArithOp::Mul: int64[idx] *= src.int64[i]; break;
						case ArithOp::Div: int64[idx] /= src.int64[i]; break;
					}
					break;
				case UINT256:
					switch (op) {
						case ArithOp::Add: uint256[idx] += src.uint256[i]; break;
						case ArithOp::Sub: uint256[idx] -= src.uint256[i]; break;
						case ArithOp::Mul: uint256[idx] = uint256[idx] * src.uint256[i]; break;
						case ArithOp::Div: uint256[idx] = uint256[idx] / src.uint256[i]; break;
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

NDArray& NDArray::broadcastOpInPlace(const NDArray& other, ArithOp op) {
	if (!isPrefixShape(other.shape, shape))
		throw std::invalid_argument("NDArray: other.shape must be a prefix of this->shape for broadcast");
	NDArrayType resultType = promoteTypes(type, other.type);
	promoteInPlace(resultType);
	if (other.type == resultType) {
		applyBroadcastInPlace(other, op);
	} else {
		NDArray tmp = other.convert(resultType);
		applyBroadcastInPlace(tmp, op);
	}
	return *this;
}

NDArray& NDArray::broadcastAdd(NDArray& other) { return broadcastOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::broadcastSub(NDArray& other) { return broadcastOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::broadcastMul(NDArray& other) { return broadcastOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::broadcastDiv(NDArray& other) { return broadcastOpInPlace(other, ArithOp::Div); }


// ---- binary operators (copy) -----------------------------------------------

NDArray NDArray::binaryOp(const NDArray& other, ArithOp op) const {
	requireSameShape(*this, other);
	NDArrayType resultType = promoteTypes(type, other.type);
	NDArray left = convert(resultType);
	NDArray right = other.convert(resultType);
	left.applyBinaryInPlace(right, op);
	return left;
}

NDArray NDArray::scalarFloatOp(float other, ArithOp op) const {
	NDArray result = convert(promoteWithFloatScalar(type));
	result.applyFloatScalarInPlace(other, op);
	return result;
}

NDArray NDArray::scalarDoubleOp(double other, ArithOp op) const {
	NDArray result = convert(promoteWithDoubleScalar(type));
	result.applyDoubleScalarInPlace(other, op);
	return result;
}

NDArray NDArray::scalarIntOp(int other, ArithOp op) const {
	NDArrayType rt = promoteWithIntScalar(type);
	NDArray result = convert(rt);
	if (rt == F32 || rt == F64)
		result.applyDoubleScalarInPlace((double)other, op);
	else
		result.applyIntScalarInPlace(other, op);
	return result;
}

NDArray NDArray::operator+(const NDArray& other) const { return binaryOp(other, ArithOp::Add); }
NDArray NDArray::operator-(const NDArray& other) const { return binaryOp(other, ArithOp::Sub); }
NDArray NDArray::operator*(const NDArray& other) const { return binaryOp(other, ArithOp::Mul); }
NDArray NDArray::operator/(const NDArray& other) const { return binaryOp(other, ArithOp::Div); }
NDArray NDArray::operator%(const NDArray& other) const { return mod(other); }

NDArray NDArray::operator+(float other) const { return scalarFloatOp(other, ArithOp::Add); }
NDArray NDArray::operator-(float other) const { return scalarFloatOp(other, ArithOp::Sub); }
NDArray NDArray::operator*(float other) const { return scalarFloatOp(other, ArithOp::Mul); }
NDArray NDArray::operator/(float other) const { return scalarFloatOp(other, ArithOp::Div); }
NDArray NDArray::operator%(float other) const { return mod(other); }

NDArray NDArray::operator+(double other) const { return scalarDoubleOp(other, ArithOp::Add); }
NDArray NDArray::operator-(double other) const { return scalarDoubleOp(other, ArithOp::Sub); }
NDArray NDArray::operator*(double other) const { return scalarDoubleOp(other, ArithOp::Mul); }
NDArray NDArray::operator/(double other) const { return scalarDoubleOp(other, ArithOp::Div); }
NDArray NDArray::operator%(double other) const { return mod(other); }

NDArray NDArray::operator+(int other) const { return scalarIntOp(other, ArithOp::Add); }
NDArray NDArray::operator-(int other) const { return scalarIntOp(other, ArithOp::Sub); }
NDArray NDArray::operator*(int other) const { return scalarIntOp(other, ArithOp::Mul); }
NDArray NDArray::operator/(int other) const { return scalarIntOp(other, ArithOp::Div); }
NDArray NDArray::operator%(int other) const { return mod(other); }

NDArray& NDArray::operator+=(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::operator-=(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::operator*=(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::operator/=(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Div); }
NDArray& NDArray::operator%=(const NDArray& other) {
	NDArray result = mod(other);
	stealFrom(result);
	return *this;
}

NDArray& NDArray::operator+=(float other) { return scalarFloatOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::operator-=(float other) { return scalarFloatOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::operator*=(float other) { return scalarFloatOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::operator/=(float other) { return scalarFloatOpInPlace(other, ArithOp::Div); }
NDArray& NDArray::operator%=(float other) {
	NDArray result = mod(other);
	stealFrom(result);
	return *this;
}

NDArray& NDArray::operator+=(double other) { return scalarDoubleOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::operator-=(double other) { return scalarDoubleOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::operator*=(double other) { return scalarDoubleOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::operator/=(double other) { return scalarDoubleOpInPlace(other, ArithOp::Div); }
NDArray& NDArray::operator%=(double other) {
	NDArray result = mod(other);
	stealFrom(result);
	return *this;
}

NDArray& NDArray::operator+=(int other) { return scalarIntOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::operator-=(int other) { return scalarIntOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::operator*=(int other) { return scalarIntOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::operator/=(int other) { return scalarIntOpInPlace(other, ArithOp::Div); }
NDArray& NDArray::operator%=(int other) {
	NDArray result = mod(other);
	stealFrom(result);
	return *this;
}


// ---- element-wise unary (IN PLACE) -----------------------------------------

NDArray& NDArray::mapRealUnary(double (*fn)(double)) {
	promoteInPlace(typeForRealUnary(type));
	const size_t n = numElements();
	if (type == F64) {
		for (size_t i = 0; i < n; ++i)
			float64[i] = fn(float64[i]);
	} else {
		// F32
		for (size_t i = 0; i < n; ++i)
			float32[i] = (float)fn((double)float32[i]);
	}
	return *this;
}

NDArray& NDArray::neg() {
	promoteInPlace(typeForNeg(type));
	const size_t n = numElements();
	switch (type) {
		case F32:
			for (size_t i = 0; i < n; ++i) float32[i] = -float32[i];
			break;
		case F64:
			for (size_t i = 0; i < n; ++i) float64[i] = -float64[i];
			break;
		case INT32:
			for (size_t i = 0; i < n; ++i) int32[i] = -int32[i];
			break;
		case INT64:
			for (size_t i = 0; i < n; ++i) int64[i] = -int64[i];
			break;
		case UINT256:
			for (size_t i = 0; i < n; ++i) uint256[i] = -uint256[i];
			break;
		default:
			throw std::runtime_error("NDArray::neg: unexpected type");
	}
	return *this;
}

NDArray NDArray::operator-() const {
	NDArray out(*this);
	out.neg();
	return out;
}

NDArray& NDArray::abs() {
	const size_t n = numElements();
	switch (type) {
		case F32:
			for (size_t i = 0; i < n; ++i) float32[i] = std::fabs(float32[i]);
			break;
		case F64:
			for (size_t i = 0; i < n; ++i) float64[i] = std::fabs(float64[i]);
			break;
		case INT32:
			for (size_t i = 0; i < n; ++i) int32[i] = int32[i] < 0 ? -int32[i] : int32[i];
			break;
		case INT64:
			for (size_t i = 0; i < n; ++i) int64[i] = int64[i] < 0 ? -int64[i] : int64[i];
			break;
		case UINT8:
		case UINT256:
		case BINARY:
			// unsigned / bits: already non-negative
			break;
		default:
			throw std::runtime_error("NDArray::abs: invalid type");
	}
	return *this;
}

NDArray& NDArray::sign() {
	const size_t n = numElements();
	switch (type) {
		case F32:
			for (size_t i = 0; i < n; ++i) {
				float v = float32[i];
				float32[i] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
			}
			break;
		case F64:
			for (size_t i = 0; i < n; ++i) {
				double v = float64[i];
				float64[i] = (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0);
			}
			break;
		case INT32:
			for (size_t i = 0; i < n; ++i)
				int32[i] = (int32[i] > 0) ? 1 : ((int32[i] < 0) ? -1 : 0);
			break;
		case INT64:
			for (size_t i = 0; i < n; ++i)
				int64[i] = (int64[i] > 0) ? 1 : ((int64[i] < 0) ? -1 : 0);
			break;
		case UINT8:
			for (size_t i = 0; i < n; ++i)
				uint8[i] = uint8[i] ? 1 : 0;
			break;
		case UINT256:
			for (size_t i = 0; i < n; ++i)
				uint256[i] = (uint256[i] == uint256_t(0)) ? uint256_t(0) : uint256_t(1);
			break;
		case BINARY:
			break;
		default:
			throw std::runtime_error("NDArray::sign: invalid type");
	}
	return *this;
}

NDArray& NDArray::square() {
	// In-place x*x; self-aliasing is fine for element-wise multiply.
	applyBinaryInPlace(*this, ArithOp::Mul);
	return *this;
}

static double d_sqrt(double x) { return std::sqrt(x); }
static double d_cbrt(double x) { return std::cbrt(x); }
static double d_exp(double x) { return std::exp(x); }
static double d_expm1(double x) { return std::expm1(x); }
static double d_log(double x) { return std::log(x); }
static double d_log2(double x) { return std::log2(x); }
static double d_log10(double x) { return std::log10(x); }
static double d_log1p(double x) { return std::log1p(x); }
static double d_sin(double x) { return std::sin(x); }
static double d_cos(double x) { return std::cos(x); }
static double d_tan(double x) { return std::tan(x); }
static double d_floor(double x) { return std::floor(x); }
static double d_ceil(double x) { return std::ceil(x); }
static double d_round(double x) { return std::round(x); }

NDArray& NDArray::sqrt() { return mapRealUnary(d_sqrt); }
NDArray& NDArray::cbrt() { return mapRealUnary(d_cbrt); }
NDArray& NDArray::exp() { return mapRealUnary(d_exp); }
NDArray& NDArray::expm1() { return mapRealUnary(d_expm1); }
NDArray& NDArray::log() { return mapRealUnary(d_log); }
NDArray& NDArray::log2() { return mapRealUnary(d_log2); }
NDArray& NDArray::log10() { return mapRealUnary(d_log10); }
NDArray& NDArray::log1p() { return mapRealUnary(d_log1p); }
NDArray& NDArray::sin() { return mapRealUnary(d_sin); }
NDArray& NDArray::cos() { return mapRealUnary(d_cos); }
NDArray& NDArray::tan() { return mapRealUnary(d_tan); }

NDArray& NDArray::floor() {
	if (!isFloatingPoint(type))
		return *this;
	return mapRealUnary(d_floor);
}
NDArray& NDArray::ceil() {
	if (!isFloatingPoint(type))
		return *this;
	return mapRealUnary(d_ceil);
}
NDArray& NDArray::round() {
	if (!isFloatingPoint(type))
		return *this;
	return mapRealUnary(d_round);
}


// ---- element-wise binary (return new) — via Impl ---------------------------

enum class ElemBinOp { Min, Max, Pow, Mod };
enum class CmpOp { Eq, Ne, Lt, Le, Gt, Ge };
enum class ReduceOp { Sum, Prod, Min, Max, Mean };

static double elemBinDouble(double a, double b, ElemBinOp op) {
	switch (op) {
		case ElemBinOp::Min: return std::fmin(a, b);
		case ElemBinOp::Max: return std::fmax(a, b);
		case ElemBinOp::Pow: return std::pow(a, b);
		case ElemBinOp::Mod: return std::fmod(a, b);
	}
	return 0.0;
}

static int64_t elemBinI64(int64_t a, int64_t b, ElemBinOp op) {
	switch (op) {
		case ElemBinOp::Min: return a < b ? a : b;
		case ElemBinOp::Max: return a > b ? a : b;
		case ElemBinOp::Pow: {
			if (b < 0)
				throw std::invalid_argument("NDArray::pow: negative exponent on integer");
			int64_t r = 1;
			int64_t base = a;
			uint64_t exp = (uint64_t)b;
			while (exp) {
				if (exp & 1ULL) r *= base;
				base *= base;
				exp >>= 1;
			}
			return r;
		}
		case ElemBinOp::Mod:
			if (b == 0)
				throw std::invalid_argument("NDArray::mod: division by zero");
			return a % b;
	}
	return 0;
}

static uint8_t elemBinU8(uint8_t a, uint8_t b, ElemBinOp op) {
	return (uint8_t)elemBinI64((int64_t)a, (int64_t)b, op);
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
				if (exp & 1ULL) r = r * base;
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

static NDArrayType promoteForElemBin(NDArrayType a, NDArrayType b, ElemBinOp op) {
	if (op == ElemBinOp::Min || op == ElemBinOp::Max) {
		if (a == b) return a;
		if (isLosslessConversion(a, b)) return b;
		if (isLosslessConversion(b, a)) return a;
		return promoteTypes(a, b);
	}
	return promoteTypes(a, b);
}

static bool cmpDouble(double a, double b, CmpOp op) {
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

static bool cmpI64(int64_t a, int64_t b, CmpOp op) {
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
					left.float32[i] = (float)elemBinDouble(left.float32[i], right.float32[i], op);
				break;
			case F64:
				for (size_t i = 0; i < n; ++i)
					left.float64[i] = elemBinDouble(left.float64[i], right.float64[i], op);
				break;
			case UINT8:
				for (size_t i = 0; i < n; ++i)
					left.uint8[i] = elemBinU8(left.uint8[i], right.uint8[i], op);
				break;
			case INT32:
				for (size_t i = 0; i < n; ++i)
					left.int32[i] = (int32_t)elemBinI64(left.int32[i], right.int32[i], op);
				break;
			case INT64:
				for (size_t i = 0; i < n; ++i)
					left.int64[i] = elemBinI64(left.int64[i], right.int64[i], op);
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

	static NDArray elemBinDoubleScalar(const NDArray& a, double s, ElemBinOp op) {
		NDArrayType rt = promoteWithDoubleScalar(a.type);
		if ((op == ElemBinOp::Min || op == ElemBinOp::Max) && isLosslessConversion(a.type, F64))
			rt = promoteTypes(a.type, F64);
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (left.type == F64) {
			for (size_t i = 0; i < n; ++i)
				left.float64[i] = elemBinDouble(left.float64[i], s, op);
		} else if (left.type == F32) {
			for (size_t i = 0; i < n; ++i)
				left.float32[i] = (float)elemBinDouble(left.float32[i], s, op);
		} else if (left.type == INT32) {
			int32_t sb = (int32_t)s;
			for (size_t i = 0; i < n; ++i)
				left.int32[i] = (int32_t)elemBinI64(left.int32[i], sb, op);
		} else if (left.type == INT64) {
			int64_t sb = (int64_t)s;
			for (size_t i = 0; i < n; ++i)
				left.int64[i] = elemBinI64(left.int64[i], sb, op);
		} else if (left.type == UINT8) {
			uint8_t sb = (uint8_t)s;
			for (size_t i = 0; i < n; ++i)
				left.uint8[i] = elemBinU8(left.uint8[i], sb, op);
		} else if (left.type == UINT256) {
			uint256_t sb(s);
			for (size_t i = 0; i < n; ++i)
				left.uint256[i] = elemBinU256(left.uint256[i], sb, op);
		} else {
			throw std::runtime_error("NDArray: invalid type in scalar element-wise op");
		}
		return left;
	}

	static NDArray elemBinFloatScalar(const NDArray& a, float s, ElemBinOp op) {
		// Prefer F32 when the array promotes there cleanly.
		NDArrayType rt = promoteWithFloatScalar(a.type);
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (left.type == F32) {
			for (size_t i = 0; i < n; ++i)
				left.float32[i] = (float)elemBinDouble(left.float32[i], s, op);
		} else {
			// Fell into a wider type (e.g. F64) — use double path on converted array
			return elemBinDoubleScalar(a, (double)s, op);
		}
		return left;
	}

	static NDArray elemBinIntScalar(const NDArray& a, int s, ElemBinOp op) {
		NDArrayType rt = promoteWithIntScalar(a.type);
		if (op == ElemBinOp::Min || op == ElemBinOp::Max) {
			if (a.type == UINT8 && s >= 0 && s <= 255) rt = UINT8;
			else if (a.type == INT32) rt = INT32;
			else if (a.type == INT64) rt = INT64;
			else if (a.type == UINT256 && s >= 0) rt = UINT256;
			else if (a.type == BINARY && (s == 0 || s == 1)) rt = BINARY;
			else if (a.type == F32) rt = F32;
			else if (a.type == F64) rt = F64;
			else rt = promoteWithIntScalar(a.type);
		}
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (left.type == F32 || left.type == F64)
			return elemBinDoubleScalar(left, (double)s, op);
		if (left.type == INT32) {
			for (size_t i = 0; i < n; ++i)
				left.int32[i] = (int32_t)elemBinI64(left.int32[i], s, op);
		} else if (left.type == INT64) {
			for (size_t i = 0; i < n; ++i)
				left.int64[i] = elemBinI64(left.int64[i], s, op);
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
		NDArrayType ct = promoteTypes(a.type, b.type);
		// Comparisons shouldn't force BINARY+BINARY → UINT8; keep BINARY if both are.
		if (a.type == BINARY && b.type == BINARY)
			ct = BINARY;

		NDArray left = a.convert(ct);
		NDArray right = b.convert(ct);
		NDArray out(a.shape, BINARY);
		const size_t n = left.numElements();

		for (size_t i = 0; i < n; ++i) {
			bool r = false;
			switch (ct) {
				case F32: r = cmpDouble(left.float32[i], right.float32[i], op); break;
				case F64: r = cmpDouble(left.float64[i], right.float64[i], op); break;
				case UINT8: r = cmpI64(left.uint8[i], right.uint8[i], op); break;
				case INT32: r = cmpI64(left.int32[i], right.int32[i], op); break;
				case INT64: r = cmpI64(left.int64[i], right.int64[i], op); break;
				case UINT256: r = cmpU256(left.uint256[i], right.uint256[i], op); break;
				case BINARY: r = cmpI64(binaryGet(left.uint64, i), binaryGet(right.uint64, i), op); break;
				default: throw std::runtime_error("NDArray: invalid type in comparison");
			}
			binarySet(out.uint64, i, r ? 1 : 0);
		}
		return out;
	}

	static NDArray compareDoubleScalar(const NDArray& a, double s, CmpOp op) {
		NDArrayType ct;
		if (isLosslessConversion(a.type, F64) || a.type == F64)
			ct = (a.type == F32) ? F32 : F64;
		else if (a.type == F32)
			ct = F32;
		else
			throw std::invalid_argument("NDArray: cannot compare this type with float scalar");

		// Prefer F32 when comparing F32 arrays to float-compatible scalars that fit
		if (a.type == F32)
			ct = F32;
		else if (isLosslessConversion(a.type, F64))
			ct = F64;
		else
			throw std::invalid_argument("NDArray: cannot compare this type with float scalar");

		NDArray left = a.convert(ct);
		NDArray out(a.shape, BINARY);
		const size_t n = left.numElements();
		for (size_t i = 0; i < n; ++i) {
			bool r = (ct == F32)
				? cmpDouble(left.float32[i], s, op)
				: cmpDouble(left.float64[i], s, op);
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
			bool t = condition.loadAsDouble(i) != 0.0;
			binarySet(mask.uint64, i, t ? 1 : 0);
		}

		NDArrayType rt = (x.type == y.type) ? x.type : promoteTypes(x.type, y.type);
		NDArray xv = x.convert(rt);
		NDArray yv = y.convert(rt);
		NDArray out(condition.shape, rt);

		for (size_t i = 0; i < n; ++i) {
			bool takeX = binaryGet(mask.uint64, i) != 0;
			switch (rt) {
				case F32: out.float32[i] = takeX ? xv.float32[i] : yv.float32[i]; break;
				case F64: out.float64[i] = takeX ? xv.float64[i] : yv.float64[i]; break;
				case UINT8: out.uint8[i] = takeX ? xv.uint8[i] : yv.uint8[i]; break;
				case INT32: out.int32[i] = takeX ? xv.int32[i] : yv.int32[i]; break;
				case INT64: out.int64[i] = takeX ? xv.int64[i] : yv.int64[i]; break;
				case UINT256: out.uint256[i] = takeX ? xv.uint256[i] : yv.uint256[i]; break;
				case BINARY:
					binarySet(out.uint64, i, takeX ? binaryGet(xv.uint64, i) : binaryGet(yv.uint64, i));
					break;
				default: throw std::runtime_error("NDArray::where: invalid type");
			}
		}
		return out;
	}

	// ---- reductions (abbreviated helpers use load/store) -------------------

	static NDArray reduceAll(const NDArray& a, ReduceOp op);
	static NDArray reduceAxis(const NDArray& a, int axis, ReduceOp op);
};

// Define reduceAll / reduceAxis outside struct for size (still as Impl members)

NDArray NDArray::Impl::reduceAll(const NDArray& a, ReduceOp op) {
	const size_t n = a.numElements();
	if (n == 0)
		throw std::invalid_argument("NDArray: cannot reduce empty array");

	if (op == ReduceOp::Mean) {
		NDArray s = reduceAll(a, ReduceOp::Sum);
		NDArrayType mt = meanResultType(a.type);
		NDArray out({}, mt);
		if (mt == F32)
			out.float32[0] = (float)(s.loadAsDouble(0) / (double)n);
		else
			out.float64[0] = s.loadAsDouble(0) / (double)n;
		return out;
	}

	if (op == ReduceOp::Min || op == ReduceOp::Max) {
		NDArray out({}, a.type);
		// seed from first element via convert of a single value
		switch (a.type) {
			case F32: {
				float acc = a.float32[0];
				for (size_t i = 1; i < n; ++i)
					acc = (op == ReduceOp::Min) ? std::fmin(acc, a.float32[i]) : std::fmax(acc, a.float32[i]);
				out.float32[0] = acc;
				break;
			}
			case F64: {
				double acc = a.float64[0];
				for (size_t i = 1; i < n; ++i)
					acc = (op == ReduceOp::Min) ? std::fmin(acc, a.float64[i]) : std::fmax(acc, a.float64[i]);
				out.float64[0] = acc;
				break;
			}
			case UINT8: {
				uint8_t acc = a.uint8[0];
				for (size_t i = 1; i < n; ++i) {
					uint8_t v = a.uint8[i];
					acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
				}
				out.uint8[0] = acc;
				break;
			}
			case INT32: {
				int32_t acc = a.int32[0];
				for (size_t i = 1; i < n; ++i) {
					int32_t v = a.int32[i];
					acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
				}
				out.int32[0] = acc;
				break;
			}
			case INT64: {
				int64_t acc = a.int64[0];
				for (size_t i = 1; i < n; ++i) {
					int64_t v = a.int64[i];
					acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
				}
				out.int64[0] = acc;
				break;
			}
			case UINT256: {
				uint256_t acc = a.uint256[0];
				for (size_t i = 1; i < n; ++i) {
					if (op == ReduceOp::Min) { if (a.uint256[i] < acc) acc = a.uint256[i]; }
					else { if (acc < a.uint256[i]) acc = a.uint256[i]; }
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
			default: throw std::runtime_error("min/max reduce: invalid type");
		}
		return out;
	}

	// Sum / Prod
	NDArrayType rt = sumProdAccumulateType(a.type);
	NDArray out({}, rt);
	if (rt == F32) {
		float acc = (op == ReduceOp::Sum) ? 0.0f : 1.0f;
		for (size_t i = 0; i < n; ++i) {
			float v = a.loadAsFloat(i);
			acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
		}
		out.float32[0] = acc;
	} else if (rt == F64) {
		double acc = (op == ReduceOp::Sum) ? 0.0 : 1.0;
		for (size_t i = 0; i < n; ++i) {
			double v = a.loadAsDouble(i);
			acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
		}
		out.float64[0] = acc;
	} else if (rt == INT64) {
		int64_t acc = (op == ReduceOp::Sum) ? 0 : 1;
		for (size_t i = 0; i < n; ++i) {
			int64_t v = a.loadAsI64(i);
			acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
		}
		out.int64[0] = acc;
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

NDArray NDArray::Impl::reduceAxis(const NDArray& a, int axis, ReduceOp op) {
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
		NDArrayType mt = meanResultType(a.type);
		if (s.type == mt && mt == F32) {
			for (size_t i = 0; i < s.numElements(); ++i)
				s.float32[i] /= (float)reduced;
			return s;
		}
		if (s.type == mt && mt == F64) {
			for (size_t i = 0; i < s.numElements(); ++i)
				s.float64[i] /= (double)reduced;
			return s;
		}
		NDArray out(outShape, mt);
		for (size_t i = 0; i < outN; ++i) {
			double v = s.loadAsDouble(i) / (double)reduced;
			if (mt == F32) out.float32[i] = (float)v;
			else out.float64[i] = v;
		}
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
				case F64: {
					double acc = a.float64[first];
					for (size_t r = 1; r < reduced; ++r) {
						double v = a.float64[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = (op == ReduceOp::Min) ? std::fmin(acc, v) : std::fmax(acc, v);
					}
					out.float64[oi] = acc;
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
				case INT32: {
					int32_t acc = a.int32[first];
					for (size_t r = 1; r < reduced; ++r) {
						int32_t v = a.int32[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
					}
					out.int32[oi] = acc;
					break;
				}
				case INT64: {
					int64_t acc = a.int64[first];
					for (size_t r = 1; r < reduced; ++r) {
						int64_t v = a.int64[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = (op == ReduceOp::Min) ? (v < acc ? v : acc) : (v > acc ? v : acc);
					}
					out.int64[oi] = acc;
					break;
				}
				case UINT256: {
					uint256_t acc = a.uint256[first];
					for (size_t r = 1; r < reduced; ++r) {
						const uint256_t& v = a.uint256[flatIndexWithAxis(a.shape, axis, oi, r)];
						if (op == ReduceOp::Min) { if (v < acc) acc = v; }
						else { if (acc < v) acc = v; }
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
				default: throw std::runtime_error("axis min/max: invalid type");
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
				float v = a.loadAsFloat(flatIndexWithAxis(a.shape, axis, oi, r));
				acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
			}
			out.float32[oi] = acc;
		} else if (rt == F64) {
			double acc = (op == ReduceOp::Sum) ? 0.0 : 1.0;
			for (size_t r = 0; r < reduced; ++r) {
				double v = a.loadAsDouble(flatIndexWithAxis(a.shape, axis, oi, r));
				acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
			}
			out.float64[oi] = acc;
		} else if (rt == INT64) {
			int64_t acc = (op == ReduceOp::Sum) ? 0 : 1;
			for (size_t r = 0; r < reduced; ++r) {
				int64_t v = a.loadAsI64(flatIndexWithAxis(a.shape, axis, oi, r));
				acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
			}
			out.int64[oi] = acc;
		} else {
			uint256_t acc = (op == ReduceOp::Sum) ? uint256_t(0) : uint256_t(1);
			for (size_t r = 0; r < reduced; ++r) {
				uint256_t v = a.loadAsU256(flatIndexWithAxis(a.shape, axis, oi, r));
				acc = (op == ReduceOp::Sum) ? (acc + v) : (acc * v);
			}
			out.uint256[oi] = acc;
		}
	}
	return out;
}


// ---- public wrappers -------------------------------------------------------

NDArray NDArray::minimum(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Min); }
NDArray NDArray::maximum(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Max); }
NDArray NDArray::pow(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Pow); }
NDArray NDArray::mod(const NDArray& other) const { return Impl::elemBinArrays(*this, other, ElemBinOp::Mod); }

NDArray NDArray::clip(const NDArray& lo, const NDArray& hi) const {
	return this->maximum(lo).minimum(hi);
}
NDArray NDArray::clip(float lo, float hi) const {
	return this->maximum(lo).minimum(hi);
}
NDArray NDArray::clip(double lo, double hi) const {
	return this->maximum(lo).minimum(hi);
}

NDArray NDArray::minimum(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Min); }
NDArray NDArray::maximum(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Max); }
NDArray NDArray::pow(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Pow); }
NDArray NDArray::mod(float other) const { return Impl::elemBinFloatScalar(*this, other, ElemBinOp::Mod); }

NDArray NDArray::minimum(double other) const { return Impl::elemBinDoubleScalar(*this, other, ElemBinOp::Min); }
NDArray NDArray::maximum(double other) const { return Impl::elemBinDoubleScalar(*this, other, ElemBinOp::Max); }
NDArray NDArray::pow(double other) const { return Impl::elemBinDoubleScalar(*this, other, ElemBinOp::Pow); }
NDArray NDArray::mod(double other) const { return Impl::elemBinDoubleScalar(*this, other, ElemBinOp::Mod); }

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

NDArray NDArray::equal(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Eq); }
NDArray NDArray::notEqual(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ne); }
NDArray NDArray::less(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Lt); }
NDArray NDArray::lessEqual(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Le); }
NDArray NDArray::greater(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Gt); }
NDArray NDArray::greaterEqual(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ge); }

NDArray NDArray::equal(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Eq); }
NDArray NDArray::notEqual(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ne); }
NDArray NDArray::less(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Lt); }
NDArray NDArray::lessEqual(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Le); }
NDArray NDArray::greater(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Gt); }
NDArray NDArray::greaterEqual(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ge); }

NDArray NDArray::where(const NDArray& condition, const NDArray& x, const NDArray& y) {
	return Impl::where(condition, x, y);
}

bool NDArray::any() const {
	const size_t n = numElements();
	for (size_t i = 0; i < n; ++i)
		if (loadAsDouble(i) != 0.0)
			return true;
	return false;
}

bool NDArray::all() const {
	const size_t n = numElements();
	if (n == 0) return true;
	for (size_t i = 0; i < n; ++i)
		if (loadAsDouble(i) == 0.0)
			return false;
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

namespace {
struct TypeRuleAnchor {
	TypeRuleAnchor() { retainTypeRuleHelpers(); }
};
static TypeRuleAnchor typeRuleAnchor;
}
