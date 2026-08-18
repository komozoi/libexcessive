
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, except software not created by me.
// If you find this software, please notify me immediately: komozoi@protonmail.com
//
//

#include "../include/NDArray.h"
#include "ndarray_int3.h"
#include "ndarray_score.h"
#include "ndarray_matmul.h"
#include "ndarray_half_kernels.h"
#include "ndarray_argext.h"
#include "ndarray_unary.h"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__AVX512F__)
#include <immintrin.h>
#define LIBEXCESSIVE_NDARRAY_AVX512 1
#else
#define LIBEXCESSIVE_NDARRAY_AVX512 0
#endif


/*
** TYPE RULES
**
** Arithmetic never silently loses information.  When two values meet, the
** result type is one that can represent both operands losslessly
** (see promoteTypes).  A scalar stays in the array type when it is
** exactly representable there.  Explicit convert() is the only place the
** programmer may request a potentially lossy cast.
**
** Unary *methods* mutate in place.  Operators return new arrays (copy).
** Same-type a+b allocates one result and writes into it (no convert() of
** either side). Mixed-type converts only the side that is not resultType.
** binaryOpInto never allocates dest; add/+= remain the in-place API.
*/

static bool isFloatingPoint(NDArrayType type) {
	return type == F16 || type == BF16 || type == F32 || type == F64;
}

static bool isHalfFloat(NDArrayType type) {
	return type == F16 || type == BF16;
}

static bool isSigned(NDArrayType type) {
	return type == F16 || type == BF16 || type == F32 || type == F64
		|| type == INT3 || type == INT8 || type == INT32 || type == INT64;
}

static bool isSignedInteger(NDArrayType type) {
	return type == INT3 || type == INT8 || type == INT32 || type == INT64;
}

static int maxPowerOfTypeIntPrecision(NDArrayType type) {
	switch (type) {
		case BINARY:  return 0;
		case INT3:    return 2; // magnitude bits; range fits in INT32
		case INT8:    return 7;
		case UINT8:   return 7;
		case INT32:   return 31;
		case INT64:   return 63;
		case F16:     return 11;
		case BF16:    return 8;
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
		case INT3:    return 2;
		case INT8:    return 7;
		case UINT8:   return 7;
		case INT32:   return 31;
		case INT64:   return 63;
		case F16:     return 16;
		case BF16:    return 128;
		case F32:     return 128;
		case F64:     return 1024;
		case UINT256: return 255;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

static bool isLosslessConversion(NDArrayType from, NDArrayType to) {
	if (from == to)
		return true;
	// Float → integer always drops fractional information.
	if (isFloatingPoint(from) && !isFloatingPoint(to))
		return false;
	// BF16 has F32's exponent; F16 does not. Mantissa bits alone lie.
	if (isFloatingPoint(from) && isFloatingPoint(to)
	    && maxPowerOfType(to) < maxPowerOfType(from))
		return false;
	if (maxPowerOfTypeIntPrecision(to) < maxPowerOfTypeIntPrecision(from))
		return false;
	if (isSigned(from) && !isSigned(to))
		return false;
	// Unsigned source does not fit a same-width signed dest (UINT8 → INT8).
	if (!isSigned(from) && isSigned(to)
	    && maxPowerOfTypeIntPrecision(to) <= maxPowerOfTypeIntPrecision(from))
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

/** True when v is exactly a value of storage type t. */
static bool int64ExactlyRepresentable(NDArrayType t, int64_t v) {
	switch (t) {
		case BINARY:  return v == 0 || v == 1;
		case INT3:    return v >= -4 && v <= 3;
		case UINT8:   return v >= 0 && v <= 255;
		case INT8:    return v >= -128 && v <= 127;
		case INT32:   return v >= (int64_t)INT32_MIN && v <= (int64_t)INT32_MAX;
		case INT64:   return true;
		case UINT256: return v >= 0;
		case F32: {
			float f = (float)v;
			return std::isfinite(f) && (int64_t)f == v;
		}
		case F64: {
			double d = (double)v;
			return std::isfinite(d) && (int64_t)d == v;
		}
		case F16:
		case BF16: {
			float f = (float)v;
			if (!std::isfinite(f) || (int64_t)f != v)
				return false;
			float h = ndarray_half::load(t, ndarray_half::store(t, f));
			return std::isfinite(h) && (int64_t)h == v;
		}
		default:
			return false;
	}
}

static bool doubleExactlyRepresentable(NDArrayType t, double v) {
	if (isFloatingPoint(t)) {
		if (!std::isfinite(v))
			return true;
		if (t == F64)
			return true;
		if (t == F32)
			return (double)(float)v == v;
		if ((double)(float)v != v)
			return false;
		float h = ndarray_half::load(t, ndarray_half::store(t, (float)v));
		return (double)h == v;
	}
	if (!std::isfinite(v) || v != std::trunc(v))
		return false;
	if (v < -9223372036854775808.0 || v >= 9223372036854775808.0)
		return t == UINT256 && v >= 0.0;
	return int64ExactlyRepresentable(t, (int64_t)v);
}

static NDArrayType promoteWithFloatScalar(NDArrayType arrayType, float scalar) {
	if (doubleExactlyRepresentable(arrayType, (double)scalar))
		return arrayType;
	if (isHalfFloat(arrayType))
		return arrayType;
	return promoteTypes(arrayType, F32);
}

static NDArrayType promoteWithDoubleScalar(NDArrayType arrayType, double scalar) {
	if (doubleExactlyRepresentable(arrayType, scalar))
		return arrayType;
	if (isHalfFloat(arrayType))
		return arrayType;
	return promoteTypes(arrayType, F64);
}

static NDArrayType promoteWithIntScalar(NDArrayType arrayType, int scalar) {
	if (int64ExactlyRepresentable(arrayType, scalar))
		return arrayType;
	if (arrayType == UINT256 || isHalfFloat(arrayType))
		return arrayType;
	return promoteTypes(arrayType, INT32);
}

static NDArrayType promoteWithInt64Scalar(NDArrayType arrayType, int64_t scalar) {
	if (int64ExactlyRepresentable(arrayType, scalar))
		return arrayType;
	if (arrayType == UINT256 || isHalfFloat(arrayType))
		return arrayType;
	return promoteTypes(arrayType, INT64);
}

/** Real unary ops (log, sin, …): stay in current float width. */
static NDArrayType typeForRealUnary(NDArrayType t) {
	if (t == F16 || t == BF16 || t == F32)
		return t;
	if (t == F64)
		return F64;
	if (t == UINT256)
		throw std::invalid_argument("NDArray: real unary op not supported on UINT256 without a wider float");
	if (t == BINARY || t == INT3 || t == UINT8 || t == INT8)
		return F32;
	return F64;
}

static bool integerLanesFitF32(const NDArray& a) {
	const size_t n = a.numElements();
	const int64_t lim = 16777216; // 2^24
	if (a.type == INT32) {
		const int32_t* p = a.data<int32_t>();
		for (size_t i = 0; i < n; ++i) {
			int32_t v = p[i];
			if (v > lim || v < -lim)
				return false;
		}
		return true;
	}
	if (a.type == INT64) {
		const int64_t* p = a.data<int64_t>();
		for (size_t i = 0; i < n; ++i) {
			int64_t v = p[i];
			if (v > lim || v < -lim)
				return false;
		}
		return true;
	}
	return true;
}

static NDArrayType typeForRealUnary(const NDArray& a) {
	if (a.type == INT32 || a.type == INT64)
		return integerLanesFitF32(a) ? F32 : F64;
	return typeForRealUnary(a.type);
}

static NDArrayType typeForNeg(NDArrayType t) {
	if (t == F16 || t == BF16 || t == F32 || t == F64 || t == INT3 || t == INT8 || t == INT32 || t == INT64)
		return t; // INT3: wrap two's-complement (neg of -4 stays -4)
	if (t == UINT256)
		return UINT256; // two's complement
	// BINARY / UINT8 → INT32 (exact for those ranges)
	return INT32;
}

static NDArrayType sumProdAccumulateType(NDArrayType t) {
	switch (t) {
		case F16:
		case BF16:
		case F32: return F32;
		case F64: return F64;

		case BINARY:
		case INT3:
		case UINT8:
		case INT8: return INT32;
		case INT32: return INT64;
		case INT64: return INT64;
		case UINT256:
			return UINT256;
		default:
			throw std::invalid_argument("Invalid type");
	}
}

static NDArrayType meanResultType(NDArrayType t) {
	if (t == F32 || t == F16 || t == BF16)
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
template<typename T>
static void addInto(T* dst, const T* a, const T* b, size_t n) {
	for (size_t i = 0; i < n; ++i)
		dst[i] = (T)(a[i] + b[i]);
}
template<typename T>
static void subInto(T* dst, const T* a, const T* b, size_t n) {
	for (size_t i = 0; i < n; ++i)
		dst[i] = (T)(a[i] - b[i]);
}
template<typename T>
static void mulInto(T* dst, const T* a, const T* b, size_t n) {
	for (size_t i = 0; i < n; ++i)
		dst[i] = (T)(a[i] * b[i]);
}
template<typename T>
static void divInto(T* dst, const T* a, const T* b, size_t n) {
	for (size_t i = 0; i < n; ++i)
		dst[i] = (T)(a[i] / b[i]);
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
	return (uint8_t)(words[i >> 6] >> (i & 63) & 1ULL);
}
static void binarySet(uint64_t* words, size_t i, uint8_t bit) {
	if (bit)
		words[i >> 6] |= 1ULL << (i & 63);
	else
		words[i >> 6] &= ~(1ULL << (i & 63));
}

enum NdBinOp { NdBinAnd = 0, NdBinOr = 1, NdBinXor = 2 };

static void ndBinaryBitOpInPlace(uint64_t* dst, const uint64_t* src, size_t nElems, NdBinOp op) {
	const size_t fullWords = nElems / 64;
	for (size_t w = 0; w < fullWords; ++w) {
		if (op == NdBinAnd) dst[w] &= src[w];
		else if (op == NdBinOr) dst[w] |= src[w];
		else dst[w] ^= src[w];
	}
	const unsigned rem = (unsigned)(nElems % 64);
	if (rem) {
		const uint64_t mask = (1ULL << rem) - 1ULL;
		uint64_t a = dst[fullWords] & mask;
		uint64_t b = src[fullWords] & mask;
		if (op == NdBinAnd) a &= b;
		else if (op == NdBinOr) a |= b;
		else a ^= b;
		dst[fullWords] = a & mask;
	}
}

static void ndBinaryBitOpWords(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t nElems, NdBinOp op) {
	const size_t fullWords = nElems / 64;
	for (size_t w = 0; w < fullWords; ++w) {
		if (op == NdBinAnd) dst[w] = a[w] & b[w];
		else if (op == NdBinOr) dst[w] = a[w] | b[w];
		else dst[w] = a[w] ^ b[w];
	}
	const unsigned rem = (unsigned)(nElems % 64);
	if (rem) {
		const uint64_t mask = (1ULL << rem) - 1ULL;
		uint64_t r;
		if (op == NdBinAnd) r = a[fullWords] & b[fullWords];
		else if (op == NdBinOr) r = a[fullWords] | b[fullWords];
		else r = a[fullWords] ^ b[fullWords];
		dst[fullWords] = r & mask;
	}
}

/** Type for a bare integer constant: INT3 when value ∈ {-1,0,1}, else INT32. */
static NDArrayType typeForIntConstant(int64_t value) {
	if (value == -1 || value == 0 || value == 1)
		return INT3;
	return INT32;
}

/** True if every element is -1, 0, or 1 (fits INT3 ternary). */
static bool allTernaryInts(const int32_t* data, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		int32_t v = data[i];
		if (v != -1 && v != 0 && v != 1)
			return false;
	}
	return true;
}

static bool allTernaryInts(const int64_t* data, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		int64_t v = data[i];
		if (v != -1 && v != 0 && v != 1)
			return false;
	}
	return true;
}

static void requireSameShape(const NDArray& a, const NDArray& b) {
	if (a.shape.size() != b.shape.size())
		throw std::invalid_argument("NDArray: shape rank mismatch");
	for (int i = 0; i < a.shape.size(); ++i)
		if (a.shape.get(i) != b.shape.get(i))
			throw std::invalid_argument("NDArray: shape mismatch");
}

static const NDArray& maybeConvert(const NDArray& src, NDArrayType resultType, NDArray& storage) {
	if (src.type == resultType)
		return src;
	storage = src.convert(resultType);
	return storage;
}




// ---- NDArrayBuffer ---------------------------------------------------------

static std::atomic<size_t> gOwnedBufferAllocs{0};

static void noteOwnedBufferAlloc() {
	gOwnedBufferAllocs.fetch_add(1, std::memory_order_relaxed);
}

NDArrayBuffer::NDArrayBuffer(size_t bytes, NDArrayType t) : byteSize(bytes), elementType(t) {
	data = malloc(bytes ? bytes : 1);
	if (bytes)
		bzero(data, bytes);
	noteOwnedBufferAlloc();
}

NDArrayBuffer::NDArrayBuffer(void* external, size_t bytes, NDArrayType t)
	: data(external), byteSize(bytes), elementType(t), ownsData(false) {}

NDArrayBuffer::NDArrayBuffer(const NDArrayBuffer& other)
	: byteSize(other.byteSize), elementType(other.elementType) {
	data = malloc(byteSize ? byteSize : 1);
	if (byteSize && other.data)
		memcpy(data, other.data, byteSize);
	noteOwnedBufferAlloc();
}

NDArrayBuffer::NDArrayBuffer(NDArrayBuffer&& other) noexcept
	: data(other.data), byteSize(other.byteSize), elementType(other.elementType),
	  ownsData(other.ownsData) {
	other.data = nullptr;
	other.byteSize = 0;
	other.ownsData = false;
}

NDArrayBuffer& NDArrayBuffer::operator=(const NDArrayBuffer& other) {
	if (this != &other) {
		if (ownsData)
			free(data);
		byteSize = other.byteSize;
		elementType = other.elementType;
		ownsData = true;
		data = malloc(byteSize ? byteSize : 1);
		if (byteSize && other.data)
			memcpy(data, other.data, byteSize);
		noteOwnedBufferAlloc();
	}
	return *this;
}

NDArrayBuffer& NDArrayBuffer::operator=(NDArrayBuffer&& other) noexcept {
	if (this != &other) {
		if (ownsData)
			free(data);
		data = other.data;
		byteSize = other.byteSize;
		elementType = other.elementType;
		ownsData = other.ownsData;
		other.data = nullptr;
		other.byteSize = 0;
		other.ownsData = false;
	}
	return *this;
}

NDArrayBuffer::~NDArrayBuffer() {
	if (ownsData)
		free(data);
	data = nullptr;
}


// ---- construction ----------------------------------------------------------

void NDArray::rebindPointers() {
	if (buffer && buffer.get()) {
		memory = buffer->data;
		memorySize = buffer->byteSize;
	} else {
		memory = nullptr;
		memorySize = 0;
	}
}

void NDArray::ensureWritable() {
	if (!buffer)
		return;
	NDArrayBuffer* raw = buffer.get();
	// Never CoW-detach a wrap: that would allocate a private copy and leave
	// the caller's mmap/stack buffer unchanged. All aliases write through.
	if (raw && !raw->ownsData) {
		rebindPointers();
		return;
	}
	// sp::mut() only detaches when pointerType()==COPY_ON_WRITE; after the first
	// detach it marks the handle SHARED, so a later NDArray copy would share a
	// SHARED buffer and mut() would mutate every alias. Detach on refcount alone
	// so CoW stays correct for NDArray regardless of the underlying SpPointerType.
	if (buffer.numReferences() > 1)
		buffer = buffer.copy(UNIQUE);
	rebindPointers();
}

static size_t mulOrThrow(size_t a, size_t b) {
	if (b != 0 && a > SIZE_MAX / b)
		throw std::invalid_argument("NDArray: size overflow");
	return a * b;
}

// Product of axes. Rank 0 → 1. Rejects negative axes and size_t overflow.
static size_t shapeElementCount(const ArrayList<int>& shape) {
	size_t n = 1;
	for (int i = 0; i < shape.size(); ++i) {
		int d = shape.get(i);
		if (d < 0)
			throw std::invalid_argument("NDArray: negative dimension");
		n = mulOrThrow(n, (size_t)d);
	}
	return n;
}

static size_t packedBufferBytes(NDArrayType t, size_t numElems) {
	switch (t) {
		case BINARY:
			if (numElems > SIZE_MAX - 63)
				throw std::invalid_argument("NDArray: size overflow");
			return (numElems + 63) / 64 * 8;
		case INT3:
			if (numElems > SIZE_MAX - (int3_kLanesPerWord - 1))
				throw std::invalid_argument("NDArray: size overflow");
			return int3_bufferBytes(numElems);
		case INT8:
		case UINT8:
			return numElems;
		case F16:
		case BF16:
			return mulOrThrow(numElems, 2);
		case INT32:
		case F32:
			return mulOrThrow(numElems, 4);
		case INT64:
		case F64:
			return mulOrThrow(numElems, 8);
		case UINT256:
			return mulOrThrow(numElems, 32);
		default:
			return numElems;
	}
}

size_t NDArray::bufferBytesFor(NDArrayType t, size_t numElems) {
	return packedBufferBytes(t, numElems);
}

static size_t wrapAlignment(NDArrayType t) {
	switch (t) {
		case BINARY:
		case INT3:
		case INT64:
		case F64:
		case UINT256:
			return 8;
		case INT32:
		case F32:
			return 4;
		case F16:
		case BF16:
			return 2;
		case INT8:
		case UINT8:
		default:
			return 1;
	}
}

sp<NDArrayBuffer> NDArray::makeWrapBuffer(void* data, size_t byteSize, const ArrayList<int>& shape, NDArrayType type) {
	const size_t n = shapeElementCount(shape);
	const size_t need = bufferBytesFor(type, n);
	if (need > 0 && data == nullptr)
		throw std::invalid_argument("NDArray wrap: null data");
	if (byteSize < need)
		throw std::invalid_argument("NDArray wrap: buffer smaller than required packed size");
	if (data && need > 0) {
		const size_t align = wrapAlignment(type);
		if (reinterpret_cast<std::uintptr_t>(data) % align != 0)
			throw std::invalid_argument("NDArray wrap: data is not aligned for element type");
	}
	return sp<NDArrayBuffer>(UNIQUE, data, byteSize, type);
}

sp<NDArrayBuffer> NDArray::makeWrapBuffer(const void* data, size_t byteSize, const ArrayList<int>& shape, NDArrayType type) {
	const size_t n = shapeElementCount(shape);
	const size_t need = bufferBytesFor(type, n);
	if (need > 0 && data == nullptr)
		throw std::invalid_argument("NDArray wrap: null data");
	if (byteSize < need)
		throw std::invalid_argument("NDArray wrap: buffer smaller than required packed size");
	if (data && need > 0) {
		const size_t align = wrapAlignment(type);
		if (reinterpret_cast<std::uintptr_t>(data) % align != 0)
			throw std::invalid_argument("NDArray wrap: data is not aligned for element type");
	}

	// This cast is OK because the buffer is returned as COW, so attempts to mutate it will force a copy.
	return sp<NDArrayBuffer>(COPY_ON_WRITE, (void*)data, byteSize, type);
}

NDArray::NDArray(ArrayList<int> shape_, NDArrayType type_, sp<NDArrayBuffer> buf)
	: shape(std::move(shape_)), type(type_), buffer(std::move(buf)), memory(nullptr) {
	rebindPointers();
}

NDArray NDArray::wrap(void* data, size_t byteSize, ArrayList<int> shape, NDArrayType type) {
	sp<NDArrayBuffer> buf = makeWrapBuffer(data, byteSize, shape, type);
	return NDArray(std::move(shape), type, std::move(buf));
}

static size_t elementsForPackedBytes(NDArrayType t, size_t bytes) {
	const size_t align = wrapAlignment(t);
	if (align != 0 && bytes % align != 0)
		throw std::invalid_argument("NDArray wrap: row stride is not aligned for element type");
	switch (t) {
		case BINARY:
			return mulOrThrow(bytes / 8, 64);
		case INT3:
			return mulOrThrow(bytes / 8, 16);
		case INT8:
		case UINT8:
			return bytes;
		case F16:
		case BF16:
			return bytes / 2;
		case INT32:
		case F32:
			return bytes / 4;
		case INT64:
		case F64:
			return bytes / 8;
		case UINT256:
			return bytes / 32;
		default:
			return bytes;
	}
}

static size_t tailElementCount(const ArrayList<int>& shape) {
	ArrayList<int> tail;
	for (int i = 1; i < shape.size(); ++i)
		tail.add(shape.get(i));
	return shapeElementCount(tail);
}

static ArrayList<size_t> denseRowMajorStrides(const ArrayList<int>& shape);

static size_t paddedWrapNeed(const ArrayList<int>& shape, NDArrayType type, size_t rowStrideBytes) {
	const size_t nRows = (size_t)shape.get(0);
	const size_t rowPacked = packedBufferBytes(type, tailElementCount(shape));
	if (rowStrideBytes < rowPacked)
		throw std::invalid_argument("NDArray wrap: row stride smaller than packed row");
	if (nRows == 0)
		return 0;
	const size_t prefix = mulOrThrow(nRows - 1, rowStrideBytes);
	if (rowPacked > SIZE_MAX - prefix)
		throw std::invalid_argument("NDArray: size overflow");
	return prefix + rowPacked;
}

static ArrayList<size_t> paddedWrapStrides(const ArrayList<int>& shape, NDArrayType type,
                                           size_t rowStrideBytes) {
	ArrayList<size_t> st = denseRowMajorStrides(shape);
	st.set(0, elementsForPackedBytes(type, rowStrideBytes));
	return st;
}

static void checkWrapPointer(const void* data, size_t byteSize, size_t need, NDArrayType type) {
	if (need > 0 && data == nullptr)
		throw std::invalid_argument("NDArray wrap: null data");
	if (byteSize < need)
		throw std::invalid_argument("NDArray wrap: buffer smaller than required packed size");
	if (data && need > 0) {
		const size_t align = wrapAlignment(type);
		if (reinterpret_cast<std::uintptr_t>(data) % align != 0)
			throw std::invalid_argument("NDArray wrap: data is not aligned for element type");
	}
}

NDArrayView NDArrayView::wrap(const void* data, size_t byteSize, ArrayList<int> shape,
                              NDArrayType type, size_t rowStrideBytes) {
	const size_t n = shapeElementCount(shape);
	const size_t dense = packedBufferBytes(type, n);
	if (shape.size() < 2) {
		if (rowStrideBytes != dense)
			throw std::invalid_argument("NDArray wrap: row stride requires rank >= 2");
		return wrap(data, byteSize, std::move(shape), type);
	}
	const size_t need = paddedWrapNeed(shape, type, rowStrideBytes);
	checkWrapPointer(data, byteSize, need, type);
	ArrayList<size_t> st = paddedWrapStrides(shape, type, rowStrideBytes);
	sp<NDArrayBuffer> buf(COPY_ON_WRITE, (void*)data, byteSize, type);
	return NDArrayView(std::move(buf), std::move(shape), std::move(st), 0, type);
}

NDArray NDArray::wrap(void* data, size_t byteSize, ArrayList<int> shape,
                      NDArrayType type, size_t rowStrideBytes) {
	const size_t n = shapeElementCount(shape);
	const size_t dense = packedBufferBytes(type, n);
	if (shape.size() < 2) {
		if (rowStrideBytes != dense)
			throw std::invalid_argument("NDArray wrap: row stride requires rank >= 2");
		return wrap(data, byteSize, std::move(shape), type);
	}
	const size_t rowPacked = packedBufferBytes(type, tailElementCount(shape));
	if (rowStrideBytes != rowPacked)
		throw std::invalid_argument("NDArray wrap: padded row stride requires NDArrayView::wrap");
	return wrap(data, byteSize, std::move(shape), type);
}

bool NDArray::ownsStorage() const {
	return buffer && buffer.get() && buffer->ownsData;
}

size_t NDArray::ownedBufferAllocCount() {
	return gOwnedBufferAllocs.load(std::memory_order_relaxed);
}

static ArrayList<size_t> denseRowMajorStrides(const ArrayList<int>& shape) {
	ArrayList<size_t> st;
	for (int i = 0; i < shape.size(); ++i)
		st.add((size_t)0);
	size_t stride = 1;
	for (int d = shape.size() - 1; d >= 0; --d) {
		st.set(d, stride);
		int dim = shape.get(d);
		stride *= (size_t)(dim > 0 ? dim : 1);
	}
	return st;
}

ArrayList<size_t> NDArray::rowMajorStrides(const ArrayList<int>& shape) {
	return denseRowMajorStrides(shape);
}

NDArray::NDArray() : shape({0}), type(F32), memory(nullptr) {
	initialize();
}

NDArray::NDArray(ArrayList<int> shape, NDArrayType type) : shape(std::move(shape)), type(type), memory(nullptr) {
	initialize();
}

NDArray NDArray::empty() {
	return empty(F32);
}

NDArray NDArray::empty(NDArrayType type) {
	return NDArray({0}, type);
}

NDArray::NDArray(const ArrayList<float>& vector) : shape({vector.size()}), type(F32), memory(nullptr) {
	initialize();
	if (memorySize)
		memcpy(float32, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<double>& vector) : shape({vector.size()}), type(F64), memory(nullptr) {
	initialize();
	if (memorySize)
		memcpy(float64, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<uint8_t>& vector) : shape({vector.size()}), type(UINT8), memory(nullptr) {
	initialize();
	if (memorySize)
		memcpy(uint8, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int8_t>& vector) : shape({vector.size()}), type(INT8), memory(nullptr) {
	initialize();
	if (memorySize)
		memcpy(int8, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int32_t>& vector)
	: shape({vector.size()}),
	  type(allTernaryInts(vector.getMemory(), (size_t)vector.size()) ? INT3 : INT32),
	  memory(nullptr) {
	initialize();
	if (type == INT3) {
		for (int i = 0; i < vector.size(); ++i)
			int3_setSigned(uint64, (size_t)i, vector.get(i));
	} else {
		if (memorySize)
			memcpy(int32, vector.getMemory(), memorySize);
	}
}

NDArray::NDArray(const ArrayList<int64_t>& vector)
	: shape({vector.size()}),
	  type(allTernaryInts(vector.getMemory(), (size_t)vector.size()) ? INT3 : INT64),
	  memory(nullptr) {
	initialize();
	if (type == INT3) {
		for (int i = 0; i < vector.size(); ++i)
			int3_setSigned(uint64, (size_t)i, (int)vector.get(i));
	} else {
		if (memorySize)
			memcpy(int64, vector.getMemory(), memorySize);
	}
}

NDArray::NDArray(const ArrayList<int>& shape, const ArrayList<float>& vector)
	: shape(shape), type(F32), memory(nullptr) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	if (memorySize)
		memcpy(float32, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int>& shape, const ArrayList<double>& vector)
	: shape(shape), type(F64), memory(nullptr) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	if (memorySize)
		memcpy(float64, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int>& shape, const ArrayList<uint8_t>& vector)
	: shape(shape), type(UINT8), memory(nullptr) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	if (memorySize)
		memcpy(uint8, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int>& shape, const ArrayList<int8_t>& vector)
	: shape(shape), type(INT8), memory(nullptr) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	if (memorySize)
		memcpy(int8, vector.getMemory(), memorySize);
}

NDArray::NDArray(const ArrayList<int>& shape, const ArrayList<int32_t>& vector)
	: shape(shape),
	  type(allTernaryInts(vector.getMemory(), (size_t)vector.size()) ? INT3 : INT32),
	  memory(nullptr) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	if (type == INT3) {
		for (int i = 0; i < vector.size(); ++i)
			int3_setSigned(uint64, (size_t)i, vector.get(i));
	} else {
		if (memorySize)
			memcpy(int32, vector.getMemory(), memorySize);
	}
}

NDArray::NDArray(const ArrayList<int>& shape, const ArrayList<int64_t>& vector)
	: shape(shape),
	  type(allTernaryInts(vector.getMemory(), (size_t)vector.size()) ? INT3 : INT64),
	  memory(nullptr) {
	size_t expectedSize = initialize();
	if (expectedSize != (size_t)vector.size())
		throw std::out_of_range("Attempted to construct NDArray with number of input elements not matching shape");
	if (type == INT3) {
		for (int i = 0; i < vector.size(); ++i)
			int3_setSigned(uint64, (size_t)i, (int)vector.get(i));
	} else {
		if (memorySize)
			memcpy(int64, vector.getMemory(), memorySize);
	}
}

NDArray::NDArray(const NDArray& other)
	: shape(other.shape), type(other.type), memory(nullptr) {
	// Always share as COPY_ON_WRITE so a subsequent write detaches even if the
	// source handle was left SHARED by an earlier sp::mut() detach.
	if (other.buffer)
		buffer = other.buffer.getWritableCopy();
	rebindPointers();
}

NDArray::NDArray(NDArray&& other) noexcept
	: shape(std::move(other.shape)), type(other.type), buffer(std::move(other.buffer)), memory(nullptr) {
	rebindPointers();
	other.shape = ArrayList({0});
	other.buffer = nullptr;
	other.memory = nullptr;
	other.memorySize = 0;
}

NDArray& NDArray::operator=(const NDArray& other) {
	if (this != &other) {
		shape = other.shape;
		type = other.type;
		if (other.buffer)
			buffer = other.buffer.getWritableCopy();
		else
			buffer = nullptr;
		rebindPointers();
	}
	return *this;
}

NDArray& NDArray::operator=(NDArray&& other) noexcept {
	if (this != &other) {
		shape = std::move(other.shape);
		type = other.type;
		buffer = std::move(other.buffer);
		rebindPointers();
		other.shape = ArrayList({0});
		other.buffer = nullptr;
		other.memory = nullptr;
		other.memorySize = 0;
	}
	return *this;
}

size_t NDArray::initialize() {
	size_t totalElements = shapeElementCount(shape);
	if (totalElements == 0) {
		buffer = nullptr;
		memory = nullptr;
		memorySize = 0;
		return 0;
	}

	memorySize = bufferBytesFor(type, totalElements);
	// UNIQUE sole owner; first NDArray copy becomes COPY_ON_WRITE via sp copy rules.
	buffer = sp<NDArrayBuffer>(UNIQUE, memorySize, type);
	rebindPointers();
	return totalElements;
}

size_t NDArray::numElements() const {
	return shapeElementCount(shape);
}

bool NDArray::isEmpty() const {
	return numElements() == 0;
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
	memory = nullptr;
	memorySize = 0;
}

size_t NDArray::computeOffset(const int* indices, int rank) const {
	if (rank != shape.size())
		throw std::out_of_range("NDArray::get - rank mismatch");
	size_t off = 0;
	size_t stride = 1;
	for (int d = rank - 1; d >= 0; --d) {
		int ix = indices[d];
		if (ix < 0 || ix >= shape.get(d))
			throw std::out_of_range("index out of bounds");
		off += (size_t)ix * stride;
		stride *= (size_t)shape.get(d);
	}
	return off;
}

size_t NDArray::computeOffset(const ArrayList<int>& indices) const {
	const int r = indices.size();
	if (r <= kMaxRank) {
		int tmp[kMaxRank] = {};
		for (int i = 0; i < r; ++i)
			tmp[i] = indices.get(i);
		return computeOffset(tmp, r);
	}
	if (r != shape.size())
		throw std::out_of_range("NDArray::get - rank mismatch");
	size_t off = 0;
	size_t stride = 1;
	for (int d = r - 1; d >= 0; --d) {
		int ix = indices.get(d);
		if (ix < 0 || ix >= shape.get(d))
			throw std::out_of_range("index out of bounds");
		off += (size_t)ix * stride;
		stride *= (size_t)shape.get(d);
	}
	return off;
}

void NDArray::stealFrom(NDArray& result) {
	shape = std::move(result.shape);
	type = result.type;
	buffer = std::move(result.buffer);
	rebindPointers();
	result.memory = nullptr;
	result.memorySize = 0;
}

void NDArray::fillFromDouble(double value) {
	ensureWritable();
	const size_t n = numElements();
	for (size_t i = 0; i < n; ++i)
		storeFromDouble(i, value);
}

void NDArray::fillFromI64(int64_t value) {
	ensureWritable();
	const size_t n = numElements();
	for (size_t i = 0; i < n; ++i)
		storeFromI64(i, value);
}

ArrayList<size_t> NDArray::strides() const {
	return rowMajorStrides(shape);
}

bool NDArray::isBroadcastableTo(const ArrayList<int>& targetShape) const {
	return view().isBroadcastableTo(targetShape);
}

NDArrayView NDArray::view() const {
	return NDArrayView(buffer, shape, rowMajorStrides(shape), 0, type);
}

NDArray NDArray::matmul(const NDArray& b) const {
	return view().matmul(b.view());
}

NDArray NDArray::matmul(const NDArrayView& b) const {
	return view().matmul(b);
}

NDArray NDArray::gemv(const NDArray& x) const {
	return view().gemv(x.view());
}

NDArray NDArray::gemv(const NDArrayView& x) const {
	return view().gemv(x);
}

NDArray NDArrayView::matmul(const NDArray& b) const {
	return matmul(b.view());
}

NDArray NDArrayView::gemv(const NDArray& x) const {
	return gemv(x.view());
}

NDArray NDArrayView::matmul(const NDArrayView& b) const {
	return ndmatmul(*this, b);
}

NDArray NDArrayView::gemv(const NDArrayView& x) const {
	return matmul(x);
}

NDArray NDArrayView::gemv(const NDArrayView& x, const NDArrayView& scales, int groupSize) const {
	return ndgemvScaled(*this, x, scales, groupSize);
}

NDArray NDArrayView::gemv(const NDArray& x, const NDArray& scales, int groupSize) const {
	return gemv(x.view(), scales.view(), groupSize);
}

NDArray NDArray::gemv(const NDArray& x, const NDArray& scales, int groupSize) const {
	return view().gemv(x.view(), scales.view(), groupSize);
}

NDArray NDArray::gemv(const NDArrayView& x, const NDArrayView& scales, int groupSize) const {
	return view().gemv(x, scales, groupSize);
}

NDArrayView NDArray::broadcastTo(const ArrayList<int>& targetShape) const {
	return view().broadcastTo(targetShape);
}

NDArrayView NDArray::reshapeView(const ArrayList<int>& newShape) const {
	return view().reshape(newShape);
}

NDArray NDArray::reshape(const ArrayList<int>& newShape) const {
	if (shapeElementCount(newShape) != numElements())
		throw std::invalid_argument("NDArray::reshape - element count mismatch");
	return NDArray(newShape, type, buffer);
}

NDArrayView NDArray::transpose() const { return view().transpose(); }
NDArrayView NDArray::swapaxes(int axisA, int axisB) const { return view().swapaxes(axisA, axisB); }
NDArrayView NDArray::permute(const ArrayList<int>& axes) const { return view().permute(axes); }
NDArrayView NDArray::slice(int axis, int index) const { return view().slice(axis, index); }
NDArrayView NDArray::row(int i) const { return view().row(i); }
NDArrayView NDArray::col(int j) const { return view().col(j); }

NDArray::Ref NDArray::operator[](int i) {
	return Ref(this, i);
}

NDArray::CRef NDArray::operator[](int i) const {
	return CRef(this, i);
}


// ---- NDArray::Ref / CRef ----------------------------------------------

NDArray::Ref::Ref(NDArray* parent, int first)
	: parent(parent), rank(1) {
	idx[0] = first;
}

NDArray::Ref NDArray::Ref::operator[](int i) const {
	if (rank >= kMaxRank)
		throw std::out_of_range("NDArray: rank exceeds kMaxRank");
	Ref next = *this;
	next.idx[next.rank++] = i;
	return next;
}

NDArray::CRef::CRef(const NDArray* parent, int first)
	: parent(parent), rank(1) {
	idx[0] = first;
}

NDArray::CRef NDArray::CRef::operator[](int i) const {
	if (rank >= kMaxRank)
		throw std::out_of_range("NDArray: rank exceeds kMaxRank");
	CRef next = *this;
	next.idx[next.rank++] = i;
	return next;
}


// ---- NDArrayView -----------------------------------------------------------

NDArrayView::NDArrayView() : shape({0}), type(F32) {}

NDArrayView::NDArrayView(sp<NDArrayBuffer> buf, ArrayList<int> shape_, ArrayList<size_t> strides_,
                         size_t offset_, NDArrayType type_)
	: shape(std::move(shape_)), strides(std::move(strides_)), offset(offset_), type(type_),
	  buffer(std::move(buf)) {}

NDArrayView NDArrayView::wrap(const void* data, size_t byteSize, ArrayList<int> shape, NDArrayType type) {
	sp<NDArrayBuffer> buf = NDArray::makeWrapBuffer(data, byteSize, shape, type);
	ArrayList<size_t> st = NDArray::rowMajorStrides(shape);
	return NDArrayView(std::move(buf), std::move(shape), std::move(st), 0, type);
}

size_t NDArrayView::numElements() const {
	return shapeElementCount(shape);
}

size_t NDArrayView::computeOffset(const ArrayList<int>& indices) const {
	if (indices.size() != shape.size())
		throw std::out_of_range("NDArrayView::computeOffset - rank mismatch");
	size_t o = offset;
	for (int d = 0; d < shape.size(); ++d) {
		int idx = indices.get(d);
		if (idx < 0 || idx >= shape.get(d))
			throw std::out_of_range("NDArrayView: index out of bounds");
		o += (size_t)idx * strides.get(d);
	}
	return o;
}

bool NDArrayView::isBroadcastableTo(const ArrayList<int>& targetShape) const {
	int ar = shape.size();
	int tr = targetShape.size();
	if (ar > tr)
		return false;
	for (int i = 0; i < tr; ++i) {
		int ti = targetShape.get(tr - 1 - i);
		int si = i < ar ? shape.get(ar - 1 - i) : 1;
		if (si != ti && si != 1)
			return false;
	}
	return true;
}

NDArrayView NDArrayView::broadcastTo(const ArrayList<int>& targetShape) const {
	shapeElementCount(targetShape);
	if (!isBroadcastableTo(targetShape))
		throw std::invalid_argument("NDArrayView::broadcastTo - shape not broadcastable");
	int ar = shape.size();
	int tr = targetShape.size();
	ArrayList<size_t> newStrides;
	for (int i = 0; i < tr; ++i)
		newStrides.add((size_t)0);
	for (int i = 0; i < tr; ++i) {
		int ti = targetShape.get(tr - 1 - i);
		int si = i < ar ? shape.get(ar - 1 - i) : 1;
		size_t oldStride = i < ar ? strides.get(ar - 1 - i) : 0;
		newStrides.set(tr - 1 - i, si == ti ? oldStride : (size_t)0);
		(void)ti;
	}
	return NDArrayView(buffer, targetShape, std::move(newStrides), offset, type);
}

NDArrayView NDArrayView::reshape(const ArrayList<int>& newShape) const {
	if (!isContiguous())
		throw std::invalid_argument("NDArrayView::reshape - view is not contiguous");
	size_t oldN = numElements();
	size_t newN = shapeElementCount(newShape);
	if (oldN != newN)
		throw std::invalid_argument("NDArrayView::reshape - element count mismatch");
	return NDArrayView(buffer, newShape, NDArray::rowMajorStrides(newShape), offset, type);
}

NDArray NDArrayView::reshapeOwned(const ArrayList<int>& newShape) const {
	if (shapeElementCount(newShape) != numElements())
		throw std::invalid_argument("NDArrayView::reshapeOwned - element count mismatch");
	const bool owned = buffer && buffer.get() && buffer.get()->ownsData;
	if (owned && isContiguous() && offset == 0)
		return NDArray(newShape, type, buffer);
	return copy().reshape(newShape);
}

NDArrayView NDArrayView::permute(const ArrayList<int>& axes) const {
	const int r = shape.size();
	if (axes.size() != r)
		throw std::invalid_argument("NDArrayView::permute - axes rank mismatch");
	ArrayList<int> used;
	for (int i = 0; i < r; ++i)
		used.add(0);
	ArrayList<int> ns;
	ArrayList<size_t> nst;
	for (int i = 0; i < r; ++i) {
		int ax = axes.get(i);
		if (ax < 0 || ax >= r)
			throw std::invalid_argument("NDArrayView::permute - axis out of range");
		if (used.get(ax))
			throw std::invalid_argument("NDArrayView::permute - duplicate axis");
		used.set(ax, 1);
		ns.add(shape.get(ax));
		nst.add(ax < strides.size() ? strides.get(ax) : (size_t)0);
	}
	return NDArrayView(buffer, std::move(ns), std::move(nst), offset, type);
}

NDArrayView NDArrayView::transpose() const {
	const int r = shape.size();
	if (r < 2)
		return *this;
	ArrayList<int> axes;
	for (int i = r - 1; i >= 0; --i)
		axes.add(i);
	return permute(axes);
}

NDArrayView NDArrayView::swapaxes(int axisA, int axisB) const {
	const int r = shape.size();
	if (axisA < 0 || axisA >= r || axisB < 0 || axisB >= r)
		throw std::invalid_argument("NDArrayView::swapaxes - axis out of range");
	if (axisA == axisB)
		return *this;
	ArrayList<int> axes;
	for (int i = 0; i < r; ++i)
		axes.add(i);
	axes.set(axisA, axisB);
	axes.set(axisB, axisA);
	return permute(axes);
}

NDArrayView NDArrayView::slice(int axis, int index) const {
	const int r = shape.size();
	if (r == 0)
		throw std::invalid_argument("NDArrayView::slice - cannot slice a scalar");
	if (axis < 0 || axis >= r)
		throw std::out_of_range("NDArrayView::slice - axis out of range");
	if (index < 0 || index >= shape.get(axis))
		throw std::out_of_range("NDArrayView::slice - index out of bounds");
	size_t newOff = offset + (size_t)index * (axis < strides.size() ? strides.get(axis) : (size_t)0);
	ArrayList<int> ns;
	ArrayList<size_t> nst;
	for (int d = 0; d < r; ++d) {
		if (d == axis)
			continue;
		ns.add(shape.get(d));
		nst.add(d < strides.size() ? strides.get(d) : (size_t)0);
	}
	return NDArrayView(buffer, std::move(ns), std::move(nst), newOff, type);
}

NDArrayView NDArrayView::row(int i) const {
	return slice(0, i);
}

NDArrayView NDArrayView::col(int j) const {
	const int r = shape.size();
	if (r == 0)
		throw std::invalid_argument("NDArrayView::col - cannot slice a scalar");
	return slice(r - 1, j);
}

static size_t unpackedElemSize(NDArrayType t) {
	switch (t) {
		case F32:
		case INT32:
			return 4;
		case F64:
		case INT64:
			return 8;
		case UINT8:
		case INT8:
			return 1;
		case F16:
		case BF16:
			return 2;
		case UINT256:
			return 32;
		default:
			return 0;
	}
}

static void viewStoreLane(NDArray& out, size_t dst, const void* data, NDArrayType t, size_t srcOff) {
	switch (t) {
		case F16:
		case BF16:
			out.setFlat(dst, ndarray_half::load(t, ((const uint16_t*)data)[srcOff]));
			break;
		case F32:
			out.setFlat(dst, ((const float*)data)[srcOff]);
			break;
		case F64:
			out.setFlat(dst, ((const double*)data)[srcOff]);
			break;
		case UINT8:
			out.setFlat(dst, ((const uint8_t*)data)[srcOff]);
			break;
		case INT8:
			out.setFlat(dst, ((const int8_t*)data)[srcOff]);
			break;
		case INT32:
			out.setFlat(dst, ((const int32_t*)data)[srcOff]);
			break;
		case INT64:
			out.setFlat(dst, ((const int64_t*)data)[srcOff]);
			break;
		case UINT256:
			out.setFlat(dst, ((const uint256_t*)data)[srcOff]);
			break;
		case BINARY:
			out.setFlat(dst, (int)binaryGet((const uint64_t*)data, srcOff));
			break;
		case INT3:
			out.setFlat(dst, int3_getSigned((const uint64_t*)data, srcOff));
			break;
		default:
			throw std::runtime_error("NDArrayView::copy: invalid type");
	}
}

NDArray NDArrayView::copy() const {
	NDArray out(shape, type);
	const size_t n = numElements();
	if (n == 0 || !sharedBuffer() || !sharedBuffer().get() || !sharedBuffer()->data)
		return out;
	const void* data = sharedBuffer()->data;
	const size_t esize = unpackedElemSize(type);
	if (isContiguous() && esize != 0) {
		memcpy(out.data(), (const char*)data + offset * esize, n * esize);
		return out;
	}

	const int rank = shape.size();
	if (rank == 0) {
		viewStoreLane(out, 0, data, type, offset);
		return out;
	}
	if (rank <= NDArray::kMaxRank) {
		int coord[NDArray::kMaxRank] = {};
		size_t srcOff = offset;
		size_t dst = 0;
		const int last = rank - 1;
		const int nLast = shape.get(last);
		const size_t stLast = last < strides.size() ? strides.get(last) : (size_t)0;
		while (true) {
			if (esize != 0 && stLast == 1 && nLast > 0) {
				memcpy((char*)out.data() + dst * esize,
				       (const char*)data + srcOff * esize,
				       (size_t)nLast * esize);
				dst += (size_t)nLast;
			} else {
				size_t o = srcOff;
				for (int j = 0; j < nLast; ++j) {
					viewStoreLane(out, dst++, data, type, o);
					o += stLast;
				}
			}
			int d = last - 1;
			for (; d >= 0; --d) {
				coord[d]++;
				srcOff += strides.get(d);
				if (coord[d] < shape.get(d))
					break;
				srcOff -= (size_t)shape.get(d) * strides.get(d);
				coord[d] = 0;
			}
			if (d < 0)
				break;
		}
		return out;
	}

	for (size_t i = 0; i < n; ++i)
		viewStoreLane(out, i, data, type, elementOffset(i));
	return out;
}


// ---- view scores: dot / L2 / Hamming ---------------------------------------
// Fast path: same storage type, packed row-major (offset may be nonzero).
// Generic path: per-element load via viewLoad*, still no length-n allocation.
// Packed ints widen into Acc (INT3 3*3 → 9). operator* wrap-mul is unchanged.

static bool viewIsPacked(const NDArrayView& v) {
	return v.isContiguous();
}

static size_t viewElemOffset(const NDArrayView& v, size_t flat) {
	return v.elementOffset(flat);
}

static void requireScoreLength(const NDArrayView& a, const NDArrayView& b, const char* what) {
	if (a.numElements() != b.numElements())
		throw std::invalid_argument(std::string(what) + ": length mismatch");
}

template <typename Acc>
static Acc accFromI64(int64_t v) {
	if constexpr (std::is_same_v<Acc, uint256_t>)
		return v < 0 ? uint256_t((int)v) : uint256_t((uint64_t)v);
	else
		return static_cast<Acc>(v);
}

template <typename Acc>
static Acc accFromDouble(double v) {
	if constexpr (std::is_same_v<Acc, uint256_t>)
		return uint256_t(v);
	else
		return static_cast<Acc>(v);
}

template <typename Acc>
static Acc accSqrt(Acc s) {
	if constexpr (std::is_same_v<Acc, uint256_t>)
		return uint256_t(std::sqrt(s.toDouble()));
	else
		return static_cast<Acc>(std::sqrt(static_cast<double>(s)));
}

enum class ScoreKind { Dot, L2Self, L2Pair };

static NDScoreOp toNDScoreOp(ScoreKind kind) {
	if (kind == ScoreKind::Dot)
		return NDScoreOp::Dot;
	if (kind == ScoreKind::L2Self)
		return NDScoreOp::L2Self;
	return NDScoreOp::L2Pair;
}

template <typename Acc>
static Acc accFromKernelF32(float v) {
	if constexpr (std::is_same_v<Acc, uint256_t>)
		return uint256_t((double)v);
	else
		return static_cast<Acc>(v);
}

template <typename Acc>
static Acc accFromKernelF64(double v) {
	if constexpr (std::is_same_v<Acc, uint256_t>)
		return uint256_t(v);
	else
		return static_cast<Acc>(v);
}

template <typename Acc>
static Acc accFromKernelI64(int64_t v) {
	return accFromI64<Acc>(v);
}

template <typename Acc>
static Acc scoreGeneric(const NDArrayView& a, const NDArrayView* b, size_t n, ScoreKind kind) {
	Acc acc{};
	for (size_t i = 0; i < n; ++i) {
		const size_t oa = viewElemOffset(a, i);
		if (kind == ScoreKind::L2Self) {
			if constexpr (std::is_floating_point_v<Acc>) {
				Acc v = accFromDouble<Acc>(ndarray_detail::viewLoadDouble(a, oa));
				acc += v * v;
			} else {
				Acc v = accFromI64<Acc>(ndarray_detail::viewLoadI64(a, oa));
				acc += v * v;
			}
			continue;
		}
		const size_t ob = viewElemOffset(*b, i);
		if constexpr (std::is_floating_point_v<Acc>) {
			Acc va = accFromDouble<Acc>(ndarray_detail::viewLoadDouble(a, oa));
			Acc vb = accFromDouble<Acc>(ndarray_detail::viewLoadDouble(*b, ob));
			if (kind == ScoreKind::Dot)
				acc += va * vb;
			else {
				Acc d = va - vb;
				acc += d * d;
			}
		} else {
			Acc va = accFromI64<Acc>(ndarray_detail::viewLoadI64(a, oa));
			Acc vb = accFromI64<Acc>(ndarray_detail::viewLoadI64(*b, ob));
			if (kind == ScoreKind::Dot)
				acc += va * vb;
			else {
				Acc d = va - vb;
				acc += d * d;
			}
		}
	}
	return acc;
}

template <typename Acc>
static Acc scorePackedSameType(const NDArrayView& a, const NDArrayView* b, size_t n, ScoreKind kind) {
	const void* da = a.sharedBuffer()->data;
	const size_t oa = a.getOffset();
	const NDArrayType t = a.getType();
	const NDScoreOp op = toNDScoreOp(kind);
	const void* db = kind == ScoreKind::L2Self ? nullptr : b->sharedBuffer()->data;
	const size_t ob = kind == ScoreKind::L2Self ? 0 : b->getOffset();
	switch (t) {
		case F16: {
			const uint16_t* pa = (const uint16_t*)da + oa;
			const uint16_t* pb = db ? (const uint16_t*)db + ob : pa;
			float s;
			if (kind == ScoreKind::Dot)
				s = ndhalf_dot_f16(pa, pb, n);
			else if (kind == ScoreKind::L2Self)
				s = ndhalf_l2self_f16(pa, n);
			else
				s = ndhalf_l2pair_f16(pa, pb, n);
			return accFromKernelF32<Acc>(s);
		}
		case BF16: {
			const uint16_t* pa = (const uint16_t*)da + oa;
			const uint16_t* pb = db ? (const uint16_t*)db + ob : pa;
			float s;
			if (kind == ScoreKind::Dot)
				s = ndhalf_dot_bf16(pa, pb, n);
			else if (kind == ScoreKind::L2Self)
				s = ndhalf_l2self_bf16(pa, n);
			else
				s = ndhalf_l2pair_bf16(pa, pb, n);
			return accFromKernelF32<Acc>(s);
		}
		case F32: {
			const float* pa = (const float*)da + oa;
			const float* pb = db ? (const float*)db + ob : pa;
			return accFromKernelF32<Acc>(ndscore_f32(pa, pb, n, op));
		}
		case F64: {
			const double* pa = (const double*)da + oa;
			const double* pb = db ? (const double*)db + ob : pa;
			return accFromKernelF64<Acc>(ndscore_f64(pa, pb, n, op));
		}
		case UINT8: {
			const uint8_t* pa = (const uint8_t*)da + oa;
			const uint8_t* pb = db ? (const uint8_t*)db + ob : pa;
			if constexpr (std::is_same_v<Acc, int64_t> || std::is_same_v<Acc, uint256_t>)
				return accFromKernelI64<Acc>(ndscore_u8_i64(pa, pb, n, op));
			return accFromKernelI64<Acc>(ndscore_u8_i32(pa, pb, n, op));
		}
		case INT8: {
			const int8_t* pa = (const int8_t*)da + oa;
			const int8_t* pb = db ? (const int8_t*)db + ob : pa;
			if constexpr (std::is_same_v<Acc, int64_t> || std::is_same_v<Acc, uint256_t>)
				return accFromKernelI64<Acc>(ndscore_i8_i64(pa, pb, n, op));
			return accFromKernelI64<Acc>(ndscore_i8_i32(pa, pb, n, op));
		}
		case INT32: {
			const int32_t* pa = (const int32_t*)da + oa;
			const int32_t* pb = db ? (const int32_t*)db + ob : pa;
			return accFromKernelI64<Acc>(ndscore_i32(pa, pb, n, op));
		}
		case INT64: {
			const int64_t* pa = (const int64_t*)da + oa;
			const int64_t* pb = db ? (const int64_t*)db + ob : pa;
			return accFromKernelI64<Acc>(ndscore_i64(pa, pb, n, op));
		}
		case INT3: {
			const uint64_t* wa = (const uint64_t*)da;
			const uint64_t* wb = db ? (const uint64_t*)db : wa;
			return accFromKernelI64<Acc>(ndscore_int3_i64(wa, oa, wb, ob, n, op));
		}
		case BINARY: {
			const uint64_t* wa = (const uint64_t*)da;
			const uint64_t* wb = db ? (const uint64_t*)db : wa;
			return accFromKernelI64<Acc>(ndscore_binary_i64(wa, oa, wb, ob, n, op));
		}
		case UINT256: {
			const uint256_t* pa = (const uint256_t*)da + oa;
			Acc acc{};
			if (kind == ScoreKind::L2Self) {
				for (size_t i = 0; i < n; ++i) {
					Acc v = accFromDouble<Acc>(pa[i].toDouble());
					if constexpr (std::is_same_v<Acc, uint256_t>)
						v = pa[i];
					acc += v * v;
				}
				return acc;
			}
			const uint256_t* pb = (const uint256_t*)b->sharedBuffer()->data + b->getOffset();
			for (size_t i = 0; i < n; ++i) {
				Acc va, vb;
				if constexpr (std::is_same_v<Acc, uint256_t>) {
					va = pa[i];
					vb = pb[i];
				} else {
					va = accFromDouble<Acc>(pa[i].toDouble());
					vb = accFromDouble<Acc>(pb[i].toDouble());
				}
				if (kind == ScoreKind::Dot)
					acc += va * vb;
				else {
					Acc d = va - vb;
					acc += d * d;
				}
			}
			return acc;
		}
		default:
			return scoreGeneric<Acc>(a, b, n, kind);
	}
}

template <typename Acc>
static Acc scoreViews(const NDArrayView& a, const NDArrayView* b, ScoreKind kind) {
	const size_t n = a.numElements();
	if (kind != ScoreKind::L2Self) {
		requireScoreLength(a, *b, kind == ScoreKind::Dot ? "NDArrayView::dot" : "NDArrayView::l2Squared");
	}
	if (n == 0)
		return Acc{};
	const bool packed = viewIsPacked(a) && a.sharedBuffer() && a.sharedBuffer().get()
	                    && (kind == ScoreKind::L2Self
	                        || (viewIsPacked(*b) && a.getType() == b->getType()));
	if (packed)
		return scorePackedSameType<Acc>(a, b, n, kind);
	return scoreGeneric<Acc>(a, b, n, kind);
}

template <typename Acc>
Acc NDArrayView::dot(const NDArrayView& other) const {
	return scoreViews<Acc>(*this, &other, ScoreKind::Dot);
}

template <typename Acc>
Acc NDArrayView::l2Squared() const {
	return scoreViews<Acc>(*this, nullptr, ScoreKind::L2Self);
}

template <typename Acc>
Acc NDArrayView::l2Squared(const NDArrayView& other) const {
	return scoreViews<Acc>(*this, &other, ScoreKind::L2Pair);
}

template <typename Acc>
Acc NDArrayView::l2Norm() const {
	return accSqrt<Acc>(l2Squared<Acc>());
}

template <typename Acc>
Acc NDArrayView::l2Norm(const NDArrayView& other) const {
	return accSqrt<Acc>(l2Squared<Acc>(other));
}

template <typename Acc>
Acc NDArrayView::hamming(const NDArrayView& other) const {
	if (getType() != BINARY || other.getType() != BINARY)
		throw std::invalid_argument("NDArrayView::hamming: both views must be BINARY");
	requireScoreLength(*this, other, "NDArrayView::hamming");
	const size_t n = numElements();
	if (n == 0)
		return Acc{};
	if (viewIsPacked(*this) && viewIsPacked(other)) {
		const uint64_t* wa = (const uint64_t*)sharedBuffer()->data;
		const uint64_t* wb = (const uint64_t*)other.sharedBuffer()->data;
		return accFromKernelI64<Acc>(ndscore_binary_i64(
			wa, getOffset(), wb, other.getOffset(), n, NDScoreOp::L2Pair));
	}
	Acc acc{};
	for (size_t i = 0; i < n; ++i) {
		const int64_t ba = ndarray_detail::viewLoadI64(*this, viewElemOffset(*this, i));
		const int64_t bb = ndarray_detail::viewLoadI64(other, viewElemOffset(other, i));
		acc += accFromI64<Acc>(ba != bb ? 1 : 0);
	}
	return acc;
}

#define LIBEXCESSIVE_INSTANTIATE_SCORE(Acc) \
	template Acc NDArrayView::dot<Acc>(const NDArrayView&) const; \
	template Acc NDArrayView::l2Squared<Acc>() const; \
	template Acc NDArrayView::l2Squared<Acc>(const NDArrayView&) const; \
	template Acc NDArrayView::l2Norm<Acc>() const; \
	template Acc NDArrayView::l2Norm<Acc>(const NDArrayView&) const; \
	template Acc NDArrayView::hamming<Acc>(const NDArrayView&) const;

LIBEXCESSIVE_INSTANTIATE_SCORE(float)
LIBEXCESSIVE_INSTANTIATE_SCORE(double)
LIBEXCESSIVE_INSTANTIATE_SCORE(int32_t)
LIBEXCESSIVE_INSTANTIATE_SCORE(int64_t)
LIBEXCESSIVE_INSTANTIATE_SCORE(uint256_t)
#undef LIBEXCESSIVE_INSTANTIATE_SCORE

// ---- typed reductions (sumAs / prodAs / meanAs) -----------------------------
// Acc is the template argument, same as dot<Acc>. Packed contiguous views walk
// typed pointers / packed words; F32/INT32 use the score-TU horizontal add.

template <typename Lane, typename Acc>
static Acc accFromLane(const Lane& v) {
	if constexpr (std::is_same_v<Lane, uint256_t>) {
		if constexpr (std::is_same_v<Acc, uint256_t>)
			return v;
		else
			return accFromDouble<Acc>(v.toDouble());
	} else if constexpr (std::is_same_v<Acc, uint256_t>) {
		if constexpr (std::is_floating_point_v<Lane>)
			return uint256_t((double)v);
		else if constexpr (std::is_signed_v<Lane>)
			return accFromI64<Acc>((int64_t)v);
		else
			return uint256_t((uint64_t)v);
	} else {
		return Acc(v);
	}
}

template <typename Lane, typename Acc>
static Acc denseSum(const Lane* a, size_t n) {
	Acc s0{}, s1{}, s2{}, s3{};
	size_t i = 0;
	for (; i + 4 <= n; i += 4) {
		s0 += accFromLane<Lane, Acc>(a[i]);
		s1 += accFromLane<Lane, Acc>(a[i + 1]);
		s2 += accFromLane<Lane, Acc>(a[i + 2]);
		s3 += accFromLane<Lane, Acc>(a[i + 3]);
	}
	Acc s = s0 + s1 + (s2 + s3);
	for (; i < n; ++i)
		s += accFromLane<Lane, Acc>(a[i]);
	return s;
}

template <typename Acc>
static void accMulEq(Acc& p, const Acc& x) {
	if constexpr (std::is_same_v<Acc, uint256_t>)
		p = p * x;
	else
		p *= x;
}

template <typename Lane, typename Acc>
static Acc denseProd(const Lane* a, size_t n) {
	Acc p0 = Acc(1), p1 = Acc(1), p2 = Acc(1), p3 = Acc(1);
	size_t i = 0;
	for (; i + 4 <= n; i += 4) {
		accMulEq(p0, accFromLane<Lane, Acc>(a[i]));
		accMulEq(p1, accFromLane<Lane, Acc>(a[i + 1]));
		accMulEq(p2, accFromLane<Lane, Acc>(a[i + 2]));
		accMulEq(p3, accFromLane<Lane, Acc>(a[i + 3]));
	}
	Acc p = p0 * p1 * (p2 * p3);
	for (; i < n; ++i)
		accMulEq(p, accFromLane<Lane, Acc>(a[i]));
	return p;
}

template <typename Acc>
static Acc accFromSigned(int v) {
	if constexpr (std::is_same_v<Acc, uint256_t>)
		return v < 0 ? uint256_t(v) : uint256_t((uint64_t)v);
	else
		return static_cast<Acc>(v);
}

template <typename Acc>
static Acc packedSum(const NDArrayView& v, size_t n) {
	const void* data = v.sharedBuffer()->data;
	const size_t o = v.getOffset();
	switch (v.getType()) {
		case F16: {
			const uint16_t* p = (const uint16_t*)data + o;
			if constexpr (std::is_same_v<Acc, float>)
				return ndhalf_sum_f16(p, n);
			return Acc(ndhalf_sum_f16(p, n));
		}
		case BF16: {
			const uint16_t* p = (const uint16_t*)data + o;
			if constexpr (std::is_same_v<Acc, float>)
				return ndhalf_sum_bf16(p, n);
			return Acc(ndhalf_sum_bf16(p, n));
		}
		case F32: {
			const float* p = (const float*)data + o;
			if constexpr (std::is_same_v<Acc, float>)
				return ndreduce_sum_f32(p, n);
			return denseSum<float, Acc>(p, n);
		}
		case F64:
			return denseSum<double, Acc>((const double*)data + o, n);
		case UINT8:
			return denseSum<uint8_t, Acc>((const uint8_t*)data + o, n);
		case INT8:
			return denseSum<int8_t, Acc>((const int8_t*)data + o, n);
		case INT32: {
			const int32_t* p = (const int32_t*)data + o;
			if constexpr (std::is_same_v<Acc, int32_t>)
				return ndreduce_sum_i32(p, n);
			return denseSum<int32_t, Acc>(p, n);
		}
		case INT64:
			return denseSum<int64_t, Acc>((const int64_t*)data + o, n);
		case INT3: {
			const uint64_t* w = (const uint64_t*)data;
			Acc s0{}, s1{}, s2{}, s3{};
			size_t i = 0;
			for (; i + 4 <= n; i += 4) {
				s0 += accFromSigned<Acc>(int3_getSigned(w, o + i));
				s1 += accFromSigned<Acc>(int3_getSigned(w, o + i + 1));
				s2 += accFromSigned<Acc>(int3_getSigned(w, o + i + 2));
				s3 += accFromSigned<Acc>(int3_getSigned(w, o + i + 3));
			}
			Acc s = s0 + s1 + (s2 + s3);
			for (; i < n; ++i)
				s += accFromSigned<Acc>(int3_getSigned(w, o + i));
			return s;
		}
		case BINARY:
			return accFromKernelI64<Acc>(ndscore_binary_i64(
				(const uint64_t*)data, o, (const uint64_t*)data, o, n, NDScoreOp::L2Self));
		case UINT256: {
			const uint256_t* p = (const uint256_t*)data + o;
			Acc s{};
			for (size_t i = 0; i < n; ++i) {
				if constexpr (std::is_same_v<Acc, uint256_t>)
					s += p[i];
				else
					s += accFromDouble<Acc>(p[i].toDouble());
			}
			return s;
		}
		default:
			break;
	}
	Acc s{};
	for (size_t i = 0; i < n; ++i)
		s += accFromDouble<Acc>(ndarray_detail::viewLoadDouble(v, v.getOffset() + i));
	return s;
}

template <typename Acc>
static Acc packedProd(const NDArrayView& v, size_t n) {
	const void* data = v.sharedBuffer()->data;
	const size_t o = v.getOffset();
	switch (v.getType()) {
		case F16:
			return Acc(ndhalf_prod_f16((const uint16_t*)data + o, n));
		case BF16:
			return Acc(ndhalf_prod_bf16((const uint16_t*)data + o, n));
		case F32:
			return denseProd<float, Acc>((const float*)data + o, n);
		case F64:
			return denseProd<double, Acc>((const double*)data + o, n);
		case UINT8:
			return denseProd<uint8_t, Acc>((const uint8_t*)data + o, n);
		case INT8:
			return denseProd<int8_t, Acc>((const int8_t*)data + o, n);
		case INT32:
			return denseProd<int32_t, Acc>((const int32_t*)data + o, n);
		case INT64:
			return denseProd<int64_t, Acc>((const int64_t*)data + o, n);
		case INT3: {
			const uint64_t* w = (const uint64_t*)data;
			Acc p = Acc(1);
			for (size_t i = 0; i < n; ++i)
				accMulEq(p, accFromSigned<Acc>(int3_getSigned(w, o + i)));
			return p;
		}
		case BINARY: {
			Acc p = Acc(1);
			const uint64_t* w = (const uint64_t*)data;
			for (size_t i = 0; i < n; ++i)
				accMulEq(p, Acc(binaryGet(w, o + i)));
			return p;
		}
		case UINT256: {
			const uint256_t* ptr = (const uint256_t*)data + o;
			Acc p = Acc(1);
			for (size_t i = 0; i < n; ++i) {
				if constexpr (std::is_same_v<Acc, uint256_t>)
					p = p * ptr[i];
				else
					accMulEq(p, accFromDouble<Acc>(ptr[i].toDouble()));
			}
			return p;
		}
		default:
			break;
	}
	Acc p = Acc(1);
	for (size_t i = 0; i < n; ++i)
		accMulEq(p, accFromDouble<Acc>(ndarray_detail::viewLoadDouble(v, v.getOffset() + i)));
	return p;
}

template <typename Acc>
static Acc genericSum(const NDArrayView& v, size_t n) {
	Acc s{};
	if constexpr (std::is_floating_point_v<Acc>) {
		for (size_t i = 0; i < n; ++i)
			s += accFromDouble<Acc>(ndarray_detail::viewLoadDouble(v, viewElemOffset(v, i)));
	} else if constexpr (std::is_same_v<Acc, uint256_t>) {
		for (size_t i = 0; i < n; ++i)
			s += ndarray_detail::viewLoadU256(v, viewElemOffset(v, i));
	} else {
		for (size_t i = 0; i < n; ++i)
			s += accFromI64<Acc>(ndarray_detail::viewLoadI64(v, viewElemOffset(v, i)));
	}
	return s;
}

template <typename Acc>
static Acc genericProd(const NDArrayView& v, size_t n) {
	Acc p = Acc(1);
	if constexpr (std::is_floating_point_v<Acc>) {
		for (size_t i = 0; i < n; ++i)
			accMulEq(p, accFromDouble<Acc>(ndarray_detail::viewLoadDouble(v, viewElemOffset(v, i))));
	} else if constexpr (std::is_same_v<Acc, uint256_t>) {
		for (size_t i = 0; i < n; ++i)
			p = p * ndarray_detail::viewLoadU256(v, viewElemOffset(v, i));
	} else {
		for (size_t i = 0; i < n; ++i)
			accMulEq(p, accFromI64<Acc>(ndarray_detail::viewLoadI64(v, viewElemOffset(v, i))));
	}
	return p;
}

template <typename Acc>
static Acc reduceViewSum(const NDArrayView& v) {
	const size_t n = v.numElements();
	if (n == 0)
		throw std::invalid_argument("NDArray: cannot reduce empty array");
	if (viewIsPacked(v) && v.sharedBuffer() && v.sharedBuffer().get())
		return packedSum<Acc>(v, n);
	return genericSum<Acc>(v, n);
}

template <typename Acc>
static Acc reduceViewProd(const NDArrayView& v) {
	const size_t n = v.numElements();
	if (n == 0)
		throw std::invalid_argument("NDArray: cannot reduce empty array");
	if (viewIsPacked(v) && v.sharedBuffer() && v.sharedBuffer().get())
		return packedProd<Acc>(v, n);
	return genericProd<Acc>(v, n);
}

template <typename Acc>
Acc NDArrayView::sumAs() const {
	return reduceViewSum<Acc>(*this);
}

template <typename Acc>
Acc NDArrayView::prodAs() const {
	return reduceViewProd<Acc>(*this);
}

template <typename Acc>
Acc NDArrayView::meanAs() const {
	const size_t n = numElements();
	Acc s = reduceViewSum<Acc>(*this);
	return s / Acc((uint64_t)n);
}

#define LIBEXCESSIVE_INSTANTIATE_REDUCE(Acc) \
	template Acc NDArrayView::sumAs<Acc>() const; \
	template Acc NDArrayView::prodAs<Acc>() const; \
	template Acc NDArrayView::meanAs<Acc>() const;

LIBEXCESSIVE_INSTANTIATE_REDUCE(float)
LIBEXCESSIVE_INSTANTIATE_REDUCE(double)
LIBEXCESSIVE_INSTANTIATE_REDUCE(int32_t)
LIBEXCESSIVE_INSTANTIATE_REDUCE(int64_t)
LIBEXCESSIVE_INSTANTIATE_REDUCE(uint32_t)
LIBEXCESSIVE_INSTANTIATE_REDUCE(uint64_t)
LIBEXCESSIVE_INSTANTIATE_REDUCE(uint256_t)
#undef LIBEXCESSIVE_INSTANTIATE_REDUCE

namespace ndarray_detail {
	static const void* viewPackedData(const NDArrayView& v) {
		if (!v.sharedBuffer() || !v.sharedBuffer().get() || !v.sharedBuffer().get()->data)
			throw std::out_of_range("NDArrayView: no buffer");
		return v.sharedBuffer().get()->data;
	}

	double viewLoadDouble(const NDArrayView& v, size_t elementOffset) {
		const void* data = viewPackedData(v);
		switch (v.getType()) {
			case BINARY: {
				const uint64_t* words = (const uint64_t*)data;
				return (double)(words[elementOffset >> 6] >> (elementOffset & 63) & 1ULL);
			}
			case INT3: {
				const uint64_t* words = (const uint64_t*)data;
				return (double)int3_getSigned(words, elementOffset);
			}
			case UINT8: return (double)((const uint8_t*)data)[elementOffset];
			case INT8: return (double)((const int8_t*)data)[elementOffset];
			case INT32: return (double)((const int32_t*)data)[elementOffset];
			case INT64: return (double)((const int64_t*)data)[elementOffset];
			case F16:
			case BF16:
				return (double)ndarray_half::load(v.getType(), ((const uint16_t*)data)[elementOffset]);
			case F32: return (double)((const float*)data)[elementOffset];
			case F64: return ((const double*)data)[elementOffset];
			case UINT256: return ((const uint256_t*)data)[elementOffset].toDouble();
			default: return 0;
		}
	}
	int64_t viewLoadI64(const NDArrayView& v, size_t elementOffset) {
		const void* data = viewPackedData(v);
		switch (v.getType()) {
			case BINARY: {
				const uint64_t* words = (const uint64_t*)data;
				return (int64_t)(words[elementOffset >> 6] >> (elementOffset & 63) & 1ULL);
			}
			case INT3: {
				const uint64_t* words = (const uint64_t*)data;
				return (int64_t)int3_getSigned(words, elementOffset);
			}
			case UINT8: return (int64_t)((const uint8_t*)data)[elementOffset];
			case INT8: return (int64_t)((const int8_t*)data)[elementOffset];
			case INT32: return (int64_t)((const int32_t*)data)[elementOffset];
			case INT64: return ((const int64_t*)data)[elementOffset];
			case F16:
			case BF16:
				return (int64_t)ndarray_half::load(v.getType(),
					((const uint16_t*)data)[elementOffset]);
			case F32: return (int64_t)((const float*)data)[elementOffset];
			case F64: return (int64_t)((const double*)data)[elementOffset];
			case UINT256:
				return (int64_t)(uint64_t)((const uint256_t*)data)[elementOffset];
			default:
				throw std::runtime_error("viewLoadI64: invalid type");
		}
	}
	uint256_t viewLoadU256(const NDArrayView& v, size_t elementOffset) {
		const void* data = viewPackedData(v);
		if (v.getType() == UINT256)
			return ((const uint256_t*)data)[elementOffset];
		return uint256_t(viewLoadDouble(v, elementOffset));
	}
}

NDArray::NDArray(int value)
	: shape({}), type(typeForIntConstant(value)), memory(nullptr) {
	initialize();
	set({}, value);
}

NDArray::NDArray(int64_t value)
	: shape({}), type(typeForIntConstant(value)), memory(nullptr) {
	initialize();
	set({}, value);
}

NDArray NDArray::zeros(const ArrayList<int>& shape) {
	return zeros(shape, INT3);
}

NDArray NDArray::zeros(const ArrayList<int>& shape, NDArrayType type) {
	return NDArray(shape, type);
}

NDArray NDArray::ones(const ArrayList<int>& shape) {
	return ones(shape, INT3);
}

NDArray NDArray::ones(const ArrayList<int>& shape, NDArrayType type) {
	NDArray out(shape, type);
	out.fillFromDouble(1.0);
	return out;
}

NDArray NDArray::full(const ArrayList<int>& shape, int value) {
	return full(shape, typeForIntConstant(value), value);
}

NDArray NDArray::full(const ArrayList<int>& shape, int64_t value) {
	return full(shape, typeForIntConstant(value), value);
}

NDArray NDArray::full(const ArrayList<int>& shape, NDArrayType type, float value) {
	NDArray out(shape, type);
	out.fillFromDouble((double)value);
	return out;
}

NDArray NDArray::full(const ArrayList<int>& shape, NDArrayType type, double value) {
	NDArray out(shape, type);
	out.fillFromDouble(value);
	return out;
}

NDArray NDArray::full(const ArrayList<int>& shape, NDArrayType type, int value) {
	NDArray out(shape, type);
	out.fillFromI64((int64_t)value);
	return out;
}

NDArray NDArray::full(const ArrayList<int>& shape, NDArrayType type, int64_t value) {
	NDArray out(shape, type);
	out.fillFromI64(value);
	return out;
}

NDArray NDArray::zerosLike(const NDArray& ref) {
	return zeros(ref.shape, ref.type);
}

NDArray NDArray::onesLike(const NDArray& ref) {
	return ones(ref.shape, ref.type);
}

NDArray NDArray::fullLike(const NDArray& ref, float value) {
	return full(ref.shape, ref.type, value);
}

NDArray NDArray::fullLike(const NDArray& ref, double value) {
	return full(ref.shape, ref.type, value);
}

NDArray NDArray::fullLike(const NDArray& ref, int value) {
	return full(ref.shape, ref.type, value);
}

NDArray NDArray::fullLike(const NDArray& ref, int64_t value) {
	return full(ref.shape, ref.type, value);
}


// ---- load / store ----------------------------------------------------------

float NDArray::loadAsFloat(size_t i) const {
	return (float)loadAsDouble(i);
}

double NDArray::loadAsDouble(size_t i) const {
	switch (type) {
		case BINARY:  return (double)binaryGet(uint64, i);
		case INT3:    return (double)int3_getSigned(uint64, i);
		case UINT8:   return (double)uint8[i];
		case INT8:    return (double)int8[i];
		case INT32:   return (double)int32[i];
		case INT64:   return (double)int64[i];
		case F16:
		case BF16:    return (double)ndarray_half::load(type, u16[i]);
		case F32:     return (double)float32[i];
		case F64:     return float64[i];
		case UINT256: return uint256[i].toDouble();
		default: throw std::runtime_error("loadAsDouble: invalid type");
	}
}

int64_t NDArray::loadAsI64(size_t i) const {
	switch (type) {
		case BINARY:  return (int64_t)binaryGet(uint64, i);
		case INT3:    return (int64_t)int3_getSigned(uint64, i);
		case UINT8:   return (int64_t)uint8[i];
		case INT8:    return (int64_t)int8[i];
		case INT32:   return (int64_t)int32[i];
		case INT64:   return int64[i];
		case F16:
		case BF16:    return (int64_t)ndarray_half::load(type, u16[i]);
		case F32:     return (int64_t)float32[i];
		case F64:     return (int64_t)float64[i];
		case UINT256: return (int64_t)(uint64_t)uint256[i];
		default: throw std::runtime_error("loadAsI64: invalid type");
	}
}

uint256_t NDArray::loadAsU256(size_t i) const {
	switch (type) {
		case BINARY:  return uint256_t((uint64_t)binaryGet(uint64, i));
		case INT3:    return uint256_t(int3_getSigned(uint64, i));
		case UINT8:   return uint256_t((uint64_t)uint8[i]);
		case INT8:    return uint256_t((int)int8[i]);
		case INT32:   return uint256_t((int)int32[i]);
		case INT64:   return uint256_t((uint64_t)int64[i]); // truncates sign bit interpretation
		case F16:
		case BF16:    return uint256_t((double)ndarray_half::load(type, u16[i]));
		case F32:     return uint256_t((double)float32[i]);
		case F64:     return uint256_t(float64[i]);
		case UINT256: return uint256[i];
		default: throw std::runtime_error("loadAsU256: invalid type");
	}
}

void NDArray::storeFromFloat(size_t i, float v) { storeFromDouble(i, (double)v); }

void NDArray::storeFromDouble(size_t i, double v) {
	ensureWritable();
	switch (type) {
		case BINARY:  binarySet(uint64, i, v > 0.0 ? 1 : 0); break;
		case INT3:    int3_setSigned(uint64, i, (int)v); break;
		case UINT8:   uint8[i] = (uint8_t)v; break;
		case INT8:    int8[i] = (int8_t)v; break;
		case INT32:   int32[i] = (int32_t)v; break;
		case INT64:   int64[i] = (int64_t)v; break;
		case F16:
		case BF16:    u16[i] = ndarray_half::store(type, (float)v); break;
		case F32:     float32[i] = (float)v; break;
		case F64:     float64[i] = v; break;
		case UINT256: uint256[i] = uint256_t(v); break;
		default: throw std::runtime_error("storeFromDouble: invalid type");
	}
}

void NDArray::storeFromI64(size_t i, int64_t v) {
	ensureWritable();
	switch (type) {
		case BINARY:  binarySet(uint64, i, v > 0 ? 1 : 0); break;
		case INT3:    int3_setSigned(uint64, i, (int)v); break;
		case UINT8:   uint8[i] = (uint8_t)v; break;
		case INT8:    int8[i] = (int8_t)v; break;
		case INT32:   int32[i] = (int32_t)v; break;
		case INT64:   int64[i] = v; break;
		case F16:
		case BF16:    u16[i] = ndarray_half::store(type, (float)v); break;
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
	ensureWritable();
	switch (type) {
		case BINARY:  binarySet(uint64, i, (uint64_t)v != 0 ? 1 : 0); break;
		case INT3:    int3_setSigned(uint64, i, (int)(int64_t)(uint64_t)v); break;
		case UINT8:   uint8[i] = (uint8_t)(uint64_t)v; break;
		case INT8:    int8[i] = (int8_t)(uint64_t)v; break;
		case INT32:   int32[i] = (int32_t)(uint64_t)v; break;
		case INT64:   int64[i] = (int64_t)(uint64_t)v; break;
		case F16:
		case BF16:    u16[i] = ndarray_half::store(type, (float)v.toDouble()); break;
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

	if (type == BINARY && newType == UINT8) {
		size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512 && defined(__AVX512BW__)
		for (; i + 64 <= n; i += 64) {
			__mmask64 m = (__mmask64)uint64[i / 64];
			_mm512_storeu_si512(result.uint8 + i, _mm512_maskz_set1_epi8(m, 1));
		}
#endif
		for (; i < n; ++i)
			result.uint8[i] = binaryGet(uint64, i);
		return result;
	}

	if ((type == F16 && newType == F32) || (type == F32 && newType == F16)
	    || (type == BF16 && newType == F32) || (type == F32 && newType == BF16)
	    || (type == F16 && newType == F64) || (type == F64 && newType == F16)
	    || (type == BF16 && newType == F64) || (type == F64 && newType == BF16)
	    || (type == F16 && newType == BF16) || (type == BF16 && newType == F16)) {
		if (type == F16 && newType == F32)
			ndhalf_f16_to_f32(result.float32, u16, n);
		else if (type == F32 && newType == F16)
			ndhalf_f32_to_f16(result.u16, float32, n);
		else if (type == BF16 && newType == F32)
			ndhalf_bf16_to_f32(result.float32, u16, n);
		else if (type == F32 && newType == BF16)
			ndhalf_f32_to_bf16(result.u16, float32, n);
		else {
			float tmp[64];
			for (size_t i = 0; i < n; ) {
				const size_t m = n - i < 64 ? n - i : 64;
				if (type == F16 && newType == F64) {
					ndhalf_f16_to_f32(tmp, u16 + i, m);
					for (size_t j = 0; j < m; ++j)
						result.float64[i + j] = (double)tmp[j];
				} else if (type == F64 && newType == F16) {
					for (size_t j = 0; j < m; ++j)
						tmp[j] = (float)float64[i + j];
					ndhalf_f32_to_f16(result.u16 + i, tmp, m);
				} else if (type == BF16 && newType == F64) {
					ndhalf_bf16_to_f32(tmp, u16 + i, m);
					for (size_t j = 0; j < m; ++j)
						result.float64[i + j] = (double)tmp[j];
				} else if (type == F64 && newType == BF16) {
					for (size_t j = 0; j < m; ++j)
						tmp[j] = (float)float64[i + j];
					ndhalf_f32_to_bf16(result.u16 + i, tmp, m);
				} else if (type == F16 && newType == BF16) {
					ndhalf_f16_to_f32(tmp, u16 + i, m);
					ndhalf_f32_to_bf16(result.u16 + i, tmp, m);
				} else {
					ndhalf_bf16_to_f32(tmp, u16 + i, m);
					ndhalf_f32_to_f16(result.u16 + i, tmp, m);
				}
				i += m;
			}
		}
	} else if (isFloatingPoint(type) || isFloatingPoint(newType)) {
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

static float arithF32(float a, float b, NDArray::ArithOp op) {
	switch (op) {
		case NDArray::ArithOp::Add: return a + b;
		case NDArray::ArithOp::Sub: return a - b;
		case NDArray::ArithOp::Mul: return a * b;
		case NDArray::ArithOp::Div: return a / b;
	}
	return 0.0f;
}

static NDHalfArith toHalfArith(NDArray::ArithOp op) {
	switch (op) {
		case NDArray::ArithOp::Add: return NDHalfArith::Add;
		case NDArray::ArithOp::Sub: return NDHalfArith::Sub;
		case NDArray::ArithOp::Mul: return NDHalfArith::Mul;
		case NDArray::ArithOp::Div: return NDHalfArith::Div;
	}
	return NDHalfArith::Add;
}

static void halfBinaryInPlace(uint16_t* dst, const uint16_t* src, size_t n,
                              NDArrayType t, NDArray::ArithOp op) {
	if (t == F16)
		ndhalf_arith_f16(dst, dst, src, n, toHalfArith(op));
	else
		ndhalf_arith_bf16(dst, dst, src, n, toHalfArith(op));
}

static void halfBinaryInto(uint16_t* dst, const uint16_t* a, const uint16_t* b, size_t n,
                           NDArrayType t, NDArray::ArithOp op) {
	if (t == F16)
		ndhalf_arith_f16(dst, a, b, n, toHalfArith(op));
	else
		ndhalf_arith_bf16(dst, a, b, n, toHalfArith(op));
}

void NDArray::applyBinaryInPlace(const NDArray& src, ArithOp op) {
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
			halfBinaryInPlace(u16, src.u16, n, type, op);
			break;
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
		case INT8:
			switch (op) {
				case ArithOp::Add: addBasic(int8, src.int8, n); break;
				case ArithOp::Sub: subBasic(int8, src.int8, n); break;
				case ArithOp::Mul: mulBasic(int8, src.int8, n); break;
				case ArithOp::Div: divBasic(int8, src.int8, n); break;
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
			if (op == ArithOp::Add || op == ArithOp::Sub)
				ndBinaryBitOpInPlace(uint64, src.uint64, n, NdBinXor);
			else if (op == ArithOp::Mul)
				ndBinaryBitOpInPlace(uint64, src.uint64, n, NdBinAnd);
			else {
				const size_t fullWords = n / 64;
				for (size_t w = 0; w < fullWords; ++w) {
					if (~src.uint64[w])
						throw std::invalid_argument("NDArray: division by zero");
				}
				const unsigned rem = (unsigned)(n % 64);
				if (rem) {
					const uint64_t keep = (1ULL << rem) - 1ULL;
					if ((~src.uint64[fullWords]) & keep)
						throw std::invalid_argument("NDArray: division by zero");
				}
			}
			break;
		case INT3:
			switch (op) {
				case ArithOp::Add: int3_add(uint64, src.uint64, n); break;
				case ArithOp::Sub: int3_sub(uint64, src.uint64, n); break;
				case ArithOp::Mul: int3_mul(uint64, src.uint64, n); break;
				case ArithOp::Div: int3_div(uint64, src.uint64, n); break;
			}
			break;
		default:
			throw std::runtime_error("NDArray: invalid type in arithmetic");
	}
}

void NDArray::applyBinaryInto(const NDArray& a, const NDArray& b, ArithOp op) {
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
			halfBinaryInto(u16, a.u16, b.u16, n, type, op);
			break;
		case F32:
			switch (op) {
				case ArithOp::Add: addInto(float32, a.float32, b.float32, n); break;
				case ArithOp::Sub: subInto(float32, a.float32, b.float32, n); break;
				case ArithOp::Mul: mulInto(float32, a.float32, b.float32, n); break;
				case ArithOp::Div: divInto(float32, a.float32, b.float32, n); break;
			}
			break;
		case F64:
			switch (op) {
				case ArithOp::Add: addInto(float64, a.float64, b.float64, n); break;
				case ArithOp::Sub: subInto(float64, a.float64, b.float64, n); break;
				case ArithOp::Mul: mulInto(float64, a.float64, b.float64, n); break;
				case ArithOp::Div: divInto(float64, a.float64, b.float64, n); break;
			}
			break;
		case UINT8:
			switch (op) {
				case ArithOp::Add: addInto(uint8, a.uint8, b.uint8, n); break;
				case ArithOp::Sub: subInto(uint8, a.uint8, b.uint8, n); break;
				case ArithOp::Mul: mulInto(uint8, a.uint8, b.uint8, n); break;
				case ArithOp::Div: divInto(uint8, a.uint8, b.uint8, n); break;
			}
			break;
		case INT8:
			switch (op) {
				case ArithOp::Add: addInto(int8, a.int8, b.int8, n); break;
				case ArithOp::Sub: subInto(int8, a.int8, b.int8, n); break;
				case ArithOp::Mul: mulInto(int8, a.int8, b.int8, n); break;
				case ArithOp::Div: divInto(int8, a.int8, b.int8, n); break;
			}
			break;
		case INT32:
			switch (op) {
				case ArithOp::Add: addInto(int32, a.int32, b.int32, n); break;
				case ArithOp::Sub: subInto(int32, a.int32, b.int32, n); break;
				case ArithOp::Mul: mulInto(int32, a.int32, b.int32, n); break;
				case ArithOp::Div: divInto(int32, a.int32, b.int32, n); break;
			}
			break;
		case INT64:
			switch (op) {
				case ArithOp::Add: addInto(int64, a.int64, b.int64, n); break;
				case ArithOp::Sub: subInto(int64, a.int64, b.int64, n); break;
				case ArithOp::Mul: mulInto(int64, a.int64, b.int64, n); break;
				case ArithOp::Div: divInto(int64, a.int64, b.int64, n); break;
			}
			break;
		case UINT256:
			switch (op) {
				case ArithOp::Add: addInto(uint256, a.uint256, b.uint256, n); break;
				case ArithOp::Sub: subInto(uint256, a.uint256, b.uint256, n); break;
				case ArithOp::Mul:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = a.uint256[i] * b.uint256[i];
					break;
				case ArithOp::Div:
					for (size_t i = 0; i < n; ++i)
						uint256[i] = a.uint256[i] / b.uint256[i];
					break;
			}
			break;
		case BINARY:
			if (op == ArithOp::Add || op == ArithOp::Sub)
				ndBinaryBitOpWords(uint64, a.uint64, b.uint64, n, NdBinXor);
			else if (op == ArithOp::Mul)
				ndBinaryBitOpWords(uint64, a.uint64, b.uint64, n, NdBinAnd);
			else {
				const size_t fullWords = n / 64;
				for (size_t w = 0; w < fullWords; ++w) {
					if (~b.uint64[w])
						throw std::invalid_argument("NDArray: division by zero");
				}
				const unsigned rem = (unsigned)(n % 64);
				if (rem) {
					const uint64_t keep = (1ULL << rem) - 1ULL;
					if ((~b.uint64[fullWords]) & keep)
						throw std::invalid_argument("NDArray: division by zero");
				}
				ndBinaryBitOpWords(uint64, a.uint64, a.uint64, n, NdBinAnd);
			}
			break;
		case INT3:
			switch (op) {
				case ArithOp::Add: int3_addInto(uint64, a.uint64, b.uint64, n); break;
				case ArithOp::Sub: int3_subInto(uint64, a.uint64, b.uint64, n); break;
				case ArithOp::Mul: int3_mulInto(uint64, a.uint64, b.uint64, n); break;
				case ArithOp::Div: int3_divInto(uint64, a.uint64, b.uint64, n); break;
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
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16: {
			float s = (float)scalar;
			if (type == F16)
				ndhalf_arith_f16_scalar(u16, u16, s, n, toHalfArith(op));
			else
				ndhalf_arith_bf16_scalar(u16, u16, s, n, toHalfArith(op));
			break;
		}
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
		case INT8: {
			int8_t s = (int8_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(int8, s, n); break;
				case ArithOp::Sub: subBasic(int8, s, n); break;
				case ArithOp::Mul: mulBasic(int8, s, n); break;
				case ArithOp::Div: divBasic(int8, s, n); break;
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
		case INT3: {
			uint8_t p = int3_encode((int)scalar);
			switch (op) {
				case ArithOp::Add: int3_addScalar(uint64, p, n); break;
				case ArithOp::Sub: int3_subScalar(uint64, p, n); break;
				case ArithOp::Mul: int3_mulScalar(uint64, p, n); break;
				case ArithOp::Div: int3_divScalar(uint64, p, n); break;
			}
			break;
		}
		default:
			throw std::runtime_error("NDArray: invalid type in scalar arithmetic");
	}
}

void NDArray::applyIntScalarInPlace(int scalar, ArithOp op) {
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
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
		case INT8: {
			int8_t s = (int8_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(int8, s, n); break;
				case ArithOp::Sub: subBasic(int8, s, n); break;
				case ArithOp::Mul: mulBasic(int8, s, n); break;
				case ArithOp::Div: divBasic(int8, s, n); break;
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
		case INT3: {
			uint8_t p = int3_encode(scalar);
			switch (op) {
				case ArithOp::Add: int3_addScalar(uint64, p, n); break;
				case ArithOp::Sub: int3_subScalar(uint64, p, n); break;
				case ArithOp::Mul: int3_mulScalar(uint64, p, n); break;
				case ArithOp::Div: int3_divScalar(uint64, p, n); break;
			}
			break;
		}
		default:
			throw std::runtime_error("NDArray: invalid type in scalar arithmetic");
	}
}

void NDArray::applyInt64ScalarInPlace(int64_t scalar, ArithOp op) {
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
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
		case INT8: {
			int8_t s = (int8_t)scalar;
			switch (op) {
				case ArithOp::Add: addBasic(int8, s, n); break;
				case ArithOp::Sub: subBasic(int8, s, n); break;
				case ArithOp::Mul: mulBasic(int8, s, n); break;
				case ArithOp::Div: divBasic(int8, s, n); break;
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
			switch (op) {
				case ArithOp::Add: addBasic(int64, scalar, n); break;
				case ArithOp::Sub: subBasic(int64, scalar, n); break;
				case ArithOp::Mul: mulBasic(int64, scalar, n); break;
				case ArithOp::Div: divBasic(int64, scalar, n); break;
			}
			break;
		}
		case UINT256: {
			uint256_t s = scalar < 0
				? uint256_t((int)scalar)
				: uint256_t((uint64_t)scalar);
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
		case INT3: {
			uint8_t p = int3_encode((int)scalar);
			switch (op) {
				case ArithOp::Add: int3_addScalar(uint64, p, n); break;
				case ArithOp::Sub: int3_subScalar(uint64, p, n); break;
				case ArithOp::Mul: int3_mulScalar(uint64, p, n); break;
				case ArithOp::Div: int3_divScalar(uint64, p, n); break;
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
	promoteInPlace(promoteWithFloatScalar(type, other));
	applyFloatScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::scalarDoubleOpInPlace(double other, ArithOp op) {
	promoteInPlace(promoteWithDoubleScalar(type, other));
	applyDoubleScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::scalarIntOpInPlace(int other, ArithOp op) {
	promoteInPlace(promoteWithIntScalar(type, other));
	if (isFloatingPoint(type))
		applyDoubleScalarInPlace((double)other, op);
	else
		applyIntScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::scalarInt64OpInPlace(int64_t other, ArithOp op) {
	promoteInPlace(promoteWithInt64Scalar(type, other));
	if (isFloatingPoint(type))
		applyDoubleScalarInPlace((double)other, op);
	else
		applyInt64ScalarInPlace(other, op);
	return *this;
}

NDArray& NDArray::add(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::sub(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::mul(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::div(const NDArray& other) { return binaryOpInPlace(other, ArithOp::Div); }

void NDArray::applyBroadcastInPlace(const NDArrayView& src, ArithOp op) {
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
			for (size_t i = 0; i < n; ++i) {
				float av = ndarray_half::load(type, u16[i]);
				float bv = (float)ndarray_detail::viewLoadDouble(src, viewElemOffset(src, i));
				u16[i] = ndarray_half::store(type, arithF32(av, bv, op));
			}
			break;
		case F32:
			for (size_t i = 0; i < n; ++i) {
				float b = (float)ndarray_detail::viewLoadDouble(src, viewElemOffset(src, i));
				switch (op) {
					case ArithOp::Add: float32[i] += b; break;
					case ArithOp::Sub: float32[i] -= b; break;
					case ArithOp::Mul: float32[i] *= b; break;
					case ArithOp::Div: float32[i] /= b; break;
				}
			}
			break;
		case F64:
			for (size_t i = 0; i < n; ++i) {
				double b = ndarray_detail::viewLoadDouble(src, viewElemOffset(src, i));
				switch (op) {
					case ArithOp::Add: float64[i] += b; break;
					case ArithOp::Sub: float64[i] -= b; break;
					case ArithOp::Mul: float64[i] *= b; break;
					case ArithOp::Div: float64[i] /= b; break;
				}
			}
			break;
		case UINT8:
			for (size_t i = 0; i < n; ++i) {
				uint8_t b = (uint8_t)ndarray_detail::viewLoadI64(src, viewElemOffset(src, i));
				switch (op) {
					case ArithOp::Add: uint8[i] = (uint8_t)(uint8[i] + b); break;
					case ArithOp::Sub: uint8[i] = (uint8_t)(uint8[i] - b); break;
					case ArithOp::Mul: uint8[i] = (uint8_t)(uint8[i] * b); break;
					case ArithOp::Div: uint8[i] = (uint8_t)(uint8[i] / b); break;
				}
			}
			break;
		case INT8:
			for (size_t i = 0; i < n; ++i) {
				int8_t b = (int8_t)ndarray_detail::viewLoadI64(src, viewElemOffset(src, i));
				switch (op) {
					case ArithOp::Add: int8[i] = (int8_t)(int8[i] + b); break;
					case ArithOp::Sub: int8[i] = (int8_t)(int8[i] - b); break;
					case ArithOp::Mul: int8[i] = (int8_t)(int8[i] * b); break;
					case ArithOp::Div: int8[i] = (int8_t)(int8[i] / b); break;
				}
			}
			break;
		case INT32:
			for (size_t i = 0; i < n; ++i) {
				int32_t b = (int32_t)ndarray_detail::viewLoadI64(src, viewElemOffset(src, i));
				switch (op) {
					case ArithOp::Add: int32[i] += b; break;
					case ArithOp::Sub: int32[i] -= b; break;
					case ArithOp::Mul: int32[i] *= b; break;
					case ArithOp::Div: int32[i] /= b; break;
				}
			}
			break;
		case INT64:
			for (size_t i = 0; i < n; ++i) {
				int64_t b = ndarray_detail::viewLoadI64(src, viewElemOffset(src, i));
				switch (op) {
					case ArithOp::Add: int64[i] += b; break;
					case ArithOp::Sub: int64[i] -= b; break;
					case ArithOp::Mul: int64[i] *= b; break;
					case ArithOp::Div: int64[i] /= b; break;
				}
			}
			break;
		case UINT256:
			for (size_t i = 0; i < n; ++i) {
				uint256_t b = ndarray_detail::viewLoadU256(src, viewElemOffset(src, i));
				switch (op) {
					case ArithOp::Add: uint256[i] += b; break;
					case ArithOp::Sub: uint256[i] -= b; break;
					case ArithOp::Mul: uint256[i] = uint256[i] * b; break;
					case ArithOp::Div: uint256[i] = uint256[i] / b; break;
				}
			}
			break;
		case BINARY:
			for (size_t i = 0; i < n; ++i) {
				uint8_t a = binaryGet(uint64, i);
				uint8_t b = (uint8_t)ndarray_detail::viewLoadI64(src, viewElemOffset(src, i));
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
		case INT3:
			for (size_t i = 0; i < n; ++i) {
				int av = int3_getSigned(uint64, i);
				int bv = (int)ndarray_detail::viewLoadI64(src, viewElemOffset(src, i));
				int r = 0;
				switch (op) {
					case ArithOp::Add: r = av + bv; break;
					case ArithOp::Sub: r = av - bv; break;
					case ArithOp::Mul: r = av * bv; break;
					case ArithOp::Div:
						if (bv == 0)
							throw std::invalid_argument("NDArray: division by zero");
						r = av / bv;
						break;
				}
				int3_setSigned(uint64, i, r);
			}
			break;
		default:
			throw std::runtime_error("NDArray: invalid type in broadcast arithmetic");
	}
}

NDArray& NDArray::broadcastOpInPlace(const NDArray& other, ArithOp op) {
	if (!other.isBroadcastableTo(shape))
		throw std::invalid_argument("NDArray: other is not broadcastable to this shape");
	NDArrayType resultType = promoteTypes(type, other.type);
	promoteInPlace(resultType);
	if (other.type == resultType) {
		applyBroadcastInPlace(other.broadcastTo(shape), op);
	} else {
		NDArray tmp = other.convert(resultType);
		applyBroadcastInPlace(tmp.broadcastTo(shape), op);
	}
	return *this;
}

NDArray& NDArray::broadcastAdd(const NDArray& other) { return broadcastOpInPlace(other, ArithOp::Add); }
NDArray& NDArray::broadcastSub(const NDArray& other) { return broadcastOpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::broadcastMul(const NDArray& other) { return broadcastOpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::broadcastDiv(const NDArray& other) { return broadcastOpInPlace(other, ArithOp::Div); }


// ---- binary operators (copy) -----------------------------------------------

void NDArray::binaryOpInto(NDArray& dst, const NDArray& a, const NDArray& b, ArithOp op) {
	requireSameShape(a, b);
	requireSameShape(dst, a);
	NDArrayType resultType = promoteTypes(a.type, b.type);
	if (dst.type != resultType)
		throw std::invalid_argument("NDArray: binaryOpInto dest type must be the promoted result type");
	NDArray aTmp, bTmp;
	const NDArray& lhs = maybeConvert(a, resultType, aTmp);
	const NDArray& rhs = maybeConvert(b, resultType, bTmp);
	dst.applyBinaryInto(lhs, rhs, op);
}

NDArray NDArray::binaryOp(const NDArray& other, ArithOp op) const {
	requireSameShape(*this, other);
	NDArrayType resultType = promoteTypes(type, other.type);
	NDArray leftTmp, rightTmp;
	const NDArray& left = maybeConvert(*this, resultType, leftTmp);
	const NDArray& right = maybeConvert(other, resultType, rightTmp);
	if (type == resultType) {
		// Left already matches: do not CoW-share *this (wrap would be mutated).
		NDArray result(shape, resultType);
		result.applyBinaryInto(left, right, op);
		return result;
	}
	// leftTmp is a unique converted buffer; reuse it as the result.
	leftTmp.applyBinaryInPlace(right, op);
	return leftTmp;
}

NDArray NDArray::scalarFloatOp(float other, ArithOp op) const {
	NDArray result = convert(promoteWithFloatScalar(type, other));
	result.applyFloatScalarInPlace(other, op);
	return result;
}

NDArray NDArray::scalarDoubleOp(double other, ArithOp op) const {
	NDArray result = convert(promoteWithDoubleScalar(type, other));
	result.applyDoubleScalarInPlace(other, op);
	return result;
}

NDArray NDArray::scalarIntOp(int other, ArithOp op) const {
	NDArrayType rt = promoteWithIntScalar(type, other);
	NDArray result = convert(rt);
	if (isFloatingPoint(rt))
		result.applyDoubleScalarInPlace((double)other, op);
	else
		result.applyIntScalarInPlace(other, op);
	return result;
}

NDArray NDArray::scalarInt64Op(int64_t other, ArithOp op) const {
	NDArrayType rt = promoteWithInt64Scalar(type, other);
	NDArray result = convert(rt);
	if (isFloatingPoint(rt))
		result.applyDoubleScalarInPlace((double)other, op);
	else
		result.applyInt64ScalarInPlace(other, op);
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

NDArray NDArray::operator+(int64_t other) const { return scalarInt64Op(other, ArithOp::Add); }
NDArray NDArray::operator-(int64_t other) const { return scalarInt64Op(other, ArithOp::Sub); }
NDArray NDArray::operator*(int64_t other) const { return scalarInt64Op(other, ArithOp::Mul); }
NDArray NDArray::operator/(int64_t other) const { return scalarInt64Op(other, ArithOp::Div); }
NDArray NDArray::operator%(int64_t other) const { return mod(other); }

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

NDArray& NDArray::operator+=(int64_t other) { return scalarInt64OpInPlace(other, ArithOp::Add); }
NDArray& NDArray::operator-=(int64_t other) { return scalarInt64OpInPlace(other, ArithOp::Sub); }
NDArray& NDArray::operator*=(int64_t other) { return scalarInt64OpInPlace(other, ArithOp::Mul); }
NDArray& NDArray::operator/=(int64_t other) { return scalarInt64OpInPlace(other, ArithOp::Div); }
NDArray& NDArray::operator%=(int64_t other) {
	NDArray result = mod(other);
	stealFrom(result);
	return *this;
}


// ---- element-wise unary (IN PLACE) -----------------------------------------

static double d_sqrt(double x);
static double d_cbrt(double x);
static double d_exp(double x);
static double d_expm1(double x);
static double d_log(double x);
static double d_log2(double x);
static double d_log10(double x);
static double d_log1p(double x);
static double d_sin(double x);
static double d_cos(double x);
static double d_tan(double x);
static double d_asin(double x);
static double d_acos(double x);
static double d_atan(double x);
static double d_sinh(double x);
static double d_cosh(double x);
static double d_tanh(double x);
static double d_asinh(double x);
static double d_acosh(double x);
static double d_atanh(double x);
static double d_deg2rad(double x);
static double d_rad2deg(double x);
static double d_floor(double x);
static double d_ceil(double x);
static double d_round(double x);

static bool classifyRealUnary(double (*fn)(double), NDUnaryOp* out) {
	if (fn == d_sqrt) { *out = NDUnaryOp::Sqrt; return true; }
	if (fn == d_cbrt) { *out = NDUnaryOp::Cbrt; return true; }
	if (fn == d_exp) { *out = NDUnaryOp::Exp; return true; }
	if (fn == d_expm1) { *out = NDUnaryOp::Expm1; return true; }
	if (fn == d_log) { *out = NDUnaryOp::Log; return true; }
	if (fn == d_log2) { *out = NDUnaryOp::Log2; return true; }
	if (fn == d_log10) { *out = NDUnaryOp::Log10; return true; }
	if (fn == d_log1p) { *out = NDUnaryOp::Log1p; return true; }
	if (fn == d_sin) { *out = NDUnaryOp::Sin; return true; }
	if (fn == d_cos) { *out = NDUnaryOp::Cos; return true; }
	if (fn == d_tan) { *out = NDUnaryOp::Tan; return true; }
	if (fn == d_asin) { *out = NDUnaryOp::Asin; return true; }
	if (fn == d_acos) { *out = NDUnaryOp::Acos; return true; }
	if (fn == d_atan) { *out = NDUnaryOp::Atan; return true; }
	if (fn == d_sinh) { *out = NDUnaryOp::Sinh; return true; }
	if (fn == d_cosh) { *out = NDUnaryOp::Cosh; return true; }
	if (fn == d_tanh) { *out = NDUnaryOp::Tanh; return true; }
	if (fn == d_asinh) { *out = NDUnaryOp::Asinh; return true; }
	if (fn == d_acosh) { *out = NDUnaryOp::Acosh; return true; }
	if (fn == d_atanh) { *out = NDUnaryOp::Atanh; return true; }
	if (fn == d_deg2rad) { *out = NDUnaryOp::Deg2Rad; return true; }
	if (fn == d_rad2deg) { *out = NDUnaryOp::Rad2Deg; return true; }
	if (fn == d_floor) { *out = NDUnaryOp::Floor; return true; }
	if (fn == d_ceil) { *out = NDUnaryOp::Ceil; return true; }
	if (fn == d_round) { *out = NDUnaryOp::Round; return true; }
	return false;
}

NDArray& NDArray::mapRealUnary(double (*fn)(double)) {
	promoteInPlace(typeForRealUnary(*this));
	ensureWritable();
	const size_t n = numElements();
	NDUnaryOp op;
	if (classifyRealUnary(fn, &op)) {
		if (type == F64)
			ndunary_f64(float64, n, op);
		else if (type == F16)
			ndunary_f16(u16, n, op);
		else if (type == BF16)
			ndunary_bf16(u16, n, op);
		else
			ndunary_f32(float32, n, op);
		return *this;
	}
	if (type == F64) {
		for (size_t i = 0; i < n; ++i)
			float64[i] = fn(float64[i]);
	} else if (type == F16 || type == BF16) {
		for (size_t i = 0; i < n; ++i)
			u16[i] = ndarray_half::store(type, (float)fn((double)ndarray_half::load(type, u16[i])));
	} else {
		for (size_t i = 0; i < n; ++i)
			float32[i] = (float)fn((double)float32[i]);
	}
	return *this;
}

NDArray& NDArray::neg() {
	promoteInPlace(typeForNeg(type));
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
			for (size_t i = 0; i < n; ++i) u16[i] = (uint16_t)(u16[i] ^ 0x8000u);
			break;
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
		case INT3:
			int3_neg(uint64, n);
			break;
		case INT8:
			for (size_t i = 0; i < n; ++i) int8[i] = (int8_t)-int8[i];
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

NDArray NDArray::squared() const {
	NDArray out(*this);
	out.square();
	return out;
}

NDArray NDArray::negated() const {
	NDArray out(*this);
	out.neg();
	return out;
}

NDArray NDArray::absolute() const {
	NDArray out(*this);
	out.abs();
	return out;
}

NDArray& NDArray::abs() {
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
			for (size_t i = 0; i < n; ++i) u16[i] = (uint16_t)(u16[i] & 0x7fffu);
			break;
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
		case INT3:
			// abs(-4) wraps to -4 in INT3; abs of other negatives fits.
			for (size_t i = 0; i < n; ++i) {
				int v = int3_getSigned(uint64, i);
				if (v < 0)
					int3_setSigned(uint64, i, -v);
			}
			break;
		case INT8:
			for (size_t i = 0; i < n; ++i) {
				int8_t v = int8[i];
				if (v < 0)
					int8[i] = (int8_t)-v;
			}
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
	ensureWritable();
	const size_t n = numElements();
	switch (type) {
		case F16:
		case BF16:
			for (size_t i = 0; i < n; ++i) {
				float v = ndarray_half::load(type, u16[i]);
				float s = v > 0.0f ? 1.0f : v < 0.0f ? -1.0f : 0.0f;
				u16[i] = ndarray_half::store(type, s);
			}
			break;
		case F32:
			for (size_t i = 0; i < n; ++i) {
				float v = float32[i];
				float32[i] = v > 0.0f ? 1.0f : v < 0.0f ? -1.0f : 0.0f;
			}
			break;
		case F64:
			for (size_t i = 0; i < n; ++i) {
				double v = float64[i];
				float64[i] = v > 0.0 ? 1.0 : v < 0.0 ? -1.0 : 0.0;
			}
			break;
		case INT32:
			for (size_t i = 0; i < n; ++i)
				int32[i] = int32[i] > 0 ? 1 : int32[i] < 0 ? -1 : 0;
			break;
		case INT64:
			for (size_t i = 0; i < n; ++i)
				int64[i] = int64[i] > 0 ? 1 : int64[i] < 0 ? -1 : 0;
			break;
		case INT8:
			for (size_t i = 0; i < n; ++i)
				int8[i] = int8[i] > 0 ? 1 : int8[i] < 0 ? (int8_t)-1 : 0;
			break;
		case INT3:
			for (size_t i = 0; i < n; ++i) {
				int v = int3_getSigned(uint64, i);
				int3_setSigned(uint64, i, v > 0 ? 1 : v < 0 ? -1 : 0);
			}
			break;
		case UINT8:
			for (size_t i = 0; i < n; ++i)
				uint8[i] = uint8[i] ? 1 : 0;
			break;
		case UINT256:
			for (size_t i = 0; i < n; ++i)
				uint256[i] = uint256[i] == uint256_t(0) ? uint256_t(0) : uint256_t(1);
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
static double d_asin(double x) { return std::asin(x); }
static double d_acos(double x) { return std::acos(x); }
static double d_atan(double x) { return std::atan(x); }
static double d_sinh(double x) { return std::sinh(x); }
static double d_cosh(double x) { return std::cosh(x); }
static double d_tanh(double x) { return std::tanh(x); }
static double d_asinh(double x) { return std::asinh(x); }
static double d_acosh(double x) { return std::acosh(x); }
static double d_atanh(double x) { return std::atanh(x); }
static double d_deg2rad(double x) { return x * (3.14159265358979323846 / 180.0); }
static double d_rad2deg(double x) { return x * (180.0 / 3.14159265358979323846); }
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
NDArray& NDArray::asin() { return mapRealUnary(d_asin); }
NDArray& NDArray::acos() { return mapRealUnary(d_acos); }
NDArray& NDArray::atan() { return mapRealUnary(d_atan); }
NDArray& NDArray::sinh() { return mapRealUnary(d_sinh); }
NDArray& NDArray::cosh() { return mapRealUnary(d_cosh); }
NDArray& NDArray::tanh() { return mapRealUnary(d_tanh); }
NDArray& NDArray::asinh() { return mapRealUnary(d_asinh); }
NDArray& NDArray::acosh() { return mapRealUnary(d_acosh); }
NDArray& NDArray::atanh() { return mapRealUnary(d_atanh); }
NDArray& NDArray::deg2rad() { return mapRealUnary(d_deg2rad); }
NDArray& NDArray::rad2deg() { return mapRealUnary(d_rad2deg); }

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

static int actNormalizeAxis(const ArrayList<int>& shape, int axis) {
	const int r = shape.size();
	if (r == 0)
		return 0;
	if (axis < 0)
		axis += r;
	if (axis < 0 || axis >= r)
		throw std::out_of_range("NDArray: axis out of range");
	return axis;
}

static void softmaxF32Row(float* row, size_t n, size_t stride) {
	float m = row[0];
	for (size_t i = 1; i < n; ++i) {
		float v = row[i * stride];
		if (v > m)
			m = v;
	}
	float s = 0.0f;
	for (size_t i = 0; i < n; ++i) {
		float e = std::exp(row[i * stride] - m);
		row[i * stride] = e;
		s += e;
	}
	const float inv = s == 0.0f ? 0.0f : 1.0f / s;
	for (size_t i = 0; i < n; ++i)
		row[i * stride] *= inv;
}

static void softmaxF64Row(double* row, size_t n, size_t stride) {
	double m = row[0];
	for (size_t i = 1; i < n; ++i) {
		double v = row[i * stride];
		if (v > m)
			m = v;
	}
	double s = 0.0;
	for (size_t i = 0; i < n; ++i) {
		double e = std::exp(row[i * stride] - m);
		row[i * stride] = e;
		s += e;
	}
	const double inv = s == 0.0 ? 0.0 : 1.0 / s;
	for (size_t i = 0; i < n; ++i)
		row[i * stride] *= inv;
}

static void actAxisGeom(const ArrayList<int>& shape, int axis,
                        size_t& outer, size_t& len, size_t& inner) {
	axis = actNormalizeAxis(shape, axis);
	len = shape.size() == 0 ? 1 : (size_t)shape.get(axis);
	inner = 1;
	for (int d = axis + 1; d < shape.size(); ++d)
		inner *= (size_t)shape.get(d);
	outer = 1;
	for (int d = 0; d < axis; ++d)
		outer *= (size_t)shape.get(d);
}

static float siluF32(float x) {
	return x / (1.0f + std::exp(-x));
}
static double siluF64(double x) {
	return x / (1.0 + std::exp(-x));
}

NDArray& NDArray::softmax() {
	return softmax(shape.size() == 0 ? 0 : shape.size() - 1);
}

NDArray& NDArray::softmax(int axis) {
	promoteInPlace(typeForRealUnary(*this));
	ensureWritable();
	if (numElements() == 0)
		return *this;
	size_t outer = 1, len = 1, inner = 1;
	actAxisGeom(shape, axis, outer, len, inner);
	if (len == 0)
		return *this;

	if (type == F32) {
		for (size_t o = 0; o < outer; ++o)
			for (size_t j = 0; j < inner; ++j)
				softmaxF32Row(float32 + o * len * inner + j, len, inner);
		return *this;
	}
	if (type == F64) {
		for (size_t o = 0; o < outer; ++o)
			for (size_t j = 0; j < inner; ++j)
				softmaxF64Row(float64 + o * len * inner + j, len, inner);
		return *this;
	}
	if (type == F16 || type == BF16) {
		ArrayList<float> buf;
		buf.resize((int)len);
		float* p = buf.getMemory();
		for (size_t o = 0; o < outer; ++o) {
			for (size_t j = 0; j < inner; ++j) {
				if (inner == 1) {
					if (type == F16)
						ndhalf_f16_to_f32(p, u16 + o * len, len);
					else
						ndhalf_bf16_to_f32(p, u16 + o * len, len);
					softmaxF32Row(p, len, 1);
					if (type == F16)
						ndhalf_f32_to_f16(u16 + o * len, p, len);
					else
						ndhalf_f32_to_bf16(u16 + o * len, p, len);
				} else {
					for (size_t k = 0; k < len; ++k)
						p[k] = ndarray_half::load(type, u16[o * len * inner + k * inner + j]);
					softmaxF32Row(p, len, 1);
					for (size_t k = 0; k < len; ++k)
						u16[o * len * inner + k * inner + j] = ndarray_half::store(type, p[k]);
				}
			}
		}
		return *this;
	}
	throw std::runtime_error("NDArray::softmax: unexpected type");
}

NDArray NDArray::softmaxed() const {
	NDArray out(*this);
	out.softmax();
	return out;
}

NDArray NDArray::softmaxed(int axis) const {
	NDArray out(*this);
	out.softmax(axis);
	return out;
}

static void loadWeightF32(const NDArray& w, float* dst, size_t len, size_t srcBase, size_t srcStride) {
	if (w.type == F32 && srcStride == 1 && w.isContiguous()) {
		const float* p = w.data<float>() + srcBase;
		for (size_t i = 0; i < len; ++i)
			dst[i] = p[i];
		return;
	}
	for (size_t i = 0; i < len; ++i)
		dst[i] = w.getFlat<float>(srcBase + i * srcStride);
}

NDArray NDArray::rmsnorm(const NDArray& weight, float eps) const {
	return rmsnorm(weight, shape.size() == 0 ? 0 : shape.size() - 1, eps);
}

NDArray NDArray::rmsnorm(const NDArray& weight, int axis, float eps) const {
	NDArray x = convert(typeForRealUnary(*this));
	x.ensureWritable();
	if (x.numElements() == 0)
		return x;
	size_t outer = 1, len = 1, inner = 1;
	actAxisGeom(x.shape, axis, outer, len, inner);
	if (len == 0)
		return x;

	const bool weight1d = (weight.shape.size() == 1 && (size_t)weight.shape.get(0) == len)
		|| (weight.shape.size() == 0 && len == 1);
	const bool weightSame = weight.shape.size() == x.shape.size();
	if (weightSame) {
		for (int d = 0; d < x.shape.size(); ++d)
			if (weight.shape.get(d) != x.shape.get(d))
				throw std::invalid_argument("NDArray::rmsnorm: weight shape must be 1-D axis length or match x");
	} else if (!weight1d) {
		throw std::invalid_argument("NDArray::rmsnorm: weight shape must be 1-D axis length or match x");
	}

	ArrayList<float> wrow;
	ArrayList<float> xrow;
	wrow.resize((int)len);
	xrow.resize((int)len);
	float* wp = wrow.getMemory();
	float* xp = xrow.getMemory();
	const float invL = 1.0f / (float)len;
	const bool last = inner == 1;

	for (size_t o = 0; o < outer; ++o) {
		for (size_t j = 0; j < inner; ++j) {
			if (weight1d)
				loadWeightF32(weight, wp, len, 0, 1);
			else
				loadWeightF32(weight, wp, len, o * len * inner + j, inner);

			if (x.type == F32 && last) {
				float* row = x.float32 + o * len;
				float ss = 0.0f;
				for (size_t k = 0; k < len; ++k)
					ss += row[k] * row[k];
				const float inv = 1.0f / std::sqrt(ss * invL + eps);
				for (size_t k = 0; k < len; ++k)
					row[k] = row[k] * wp[k] * inv;
			} else if (x.type == F64 && last) {
				double* row = x.float64 + o * len;
				double ss = 0.0;
				for (size_t k = 0; k < len; ++k)
					ss += row[k] * row[k];
				const double inv = 1.0 / std::sqrt(ss / (double)len + (double)eps);
				for (size_t k = 0; k < len; ++k)
					row[k] = row[k] * (double)wp[k] * inv;
			} else {
				for (size_t k = 0; k < len; ++k)
					xp[k] = (float)x.loadAsDouble(o * len * inner + k * inner + j);
				float ss = 0.0f;
				for (size_t k = 0; k < len; ++k)
					ss += xp[k] * xp[k];
				const float inv = 1.0f / std::sqrt(ss * invL + eps);
				for (size_t k = 0; k < len; ++k)
					x.storeFromDouble(o * len * inner + k * inner + j, (double)(xp[k] * wp[k] * inv));
			}
		}
	}
	return x;
}

NDArray& NDArray::silu() {
	promoteInPlace(typeForRealUnary(*this));
	ensureWritable();
	const size_t n = numElements();
	if (type == F32) {
		for (size_t i = 0; i < n; ++i)
			float32[i] = siluF32(float32[i]);
	} else if (type == F64) {
		for (size_t i = 0; i < n; ++i)
			float64[i] = siluF64(float64[i]);
	} else if (type == F16 || type == BF16) {
		float buf[32];
		for (size_t i = 0; i < n; ) {
			const size_t m = n - i < 32 ? n - i : 32;
			if (type == F16)
				ndhalf_f16_to_f32(buf, u16 + i, m);
			else
				ndhalf_bf16_to_f32(buf, u16 + i, m);
			for (size_t j = 0; j < m; ++j)
				buf[j] = siluF32(buf[j]);
			if (type == F16)
				ndhalf_f32_to_f16(u16 + i, buf, m);
			else
				ndhalf_f32_to_bf16(u16 + i, buf, m);
			i += m;
		}
	} else {
		throw std::runtime_error("NDArray::silu: unexpected type");
	}
	return *this;
}

NDArray NDArray::silued() const {
	NDArray out(*this);
	out.silu();
	return out;
}

NDArray& NDArray::siluMul(const NDArray& other) {
	requireSameShape(*this, other);
	promoteInPlace(promoteTypes(typeForRealUnary(*this), typeForRealUnary(other)));
	NDArray rhs = other.convert(type);
	ensureWritable();
	const size_t n = numElements();
	if (type == F32) {
		for (size_t i = 0; i < n; ++i)
			float32[i] = siluF32(float32[i]) * rhs.float32[i];
	} else if (type == F64) {
		for (size_t i = 0; i < n; ++i)
			float64[i] = siluF64(float64[i]) * rhs.float64[i];
	} else if (type == F16 || type == BF16) {
		float xb[32], yb[32];
		for (size_t i = 0; i < n; ) {
			const size_t m = n - i < 32 ? n - i : 32;
			if (type == F16) {
				ndhalf_f16_to_f32(xb, u16 + i, m);
				ndhalf_f16_to_f32(yb, rhs.u16 + i, m);
			} else {
				ndhalf_bf16_to_f32(xb, u16 + i, m);
				ndhalf_bf16_to_f32(yb, rhs.u16 + i, m);
			}
			for (size_t j = 0; j < m; ++j)
				xb[j] = siluF32(xb[j]) * yb[j];
			if (type == F16)
				ndhalf_f32_to_f16(u16 + i, xb, m);
			else
				ndhalf_f32_to_bf16(u16 + i, xb, m);
			i += m;
		}
	} else {
		throw std::runtime_error("NDArray::siluMul: unexpected type");
	}
	return *this;
}

// ---- binary real functions (return new arrays) -----------------------------

static void mapRealBinary(NDArray& a, const NDArray& b, double (*fn)(double, double)) {
	const size_t n = a.numElements();
	if (a.type == F16 || a.type == BF16) {
		uint16_t* dst = a.data<uint16_t>();
		const uint16_t* src = b.data<uint16_t>();
		for (size_t i = 0; i < n; ++i) {
			double av = ndarray_half::load(a.type, dst[i]);
			double bv = ndarray_half::load(b.type, src[i]);
			dst[i] = ndarray_half::store(a.type, (float)fn(av, bv));
		}
	} else if (a.type == F32) {
		float* dst = a.data<float>();
		const float* src = b.data<float>();
		for (size_t i = 0; i < n; ++i)
			dst[i] = (float)fn((double)dst[i], (double)src[i]);
	} else {
		double* dst = a.data<double>();
		const double* src = b.data<double>();
		for (size_t i = 0; i < n; ++i)
			dst[i] = fn(dst[i], src[i]);
	}
}

static void mapRealBinaryScalar(NDArray& a, double x, double (*fn)(double, double)) {
	const size_t n = a.numElements();
	if (a.type == F16 || a.type == BF16) {
		uint16_t* dst = a.data<uint16_t>();
		for (size_t i = 0; i < n; ++i) {
			double av = ndarray_half::load(a.type, dst[i]);
			dst[i] = ndarray_half::store(a.type, (float)fn(av, x));
		}
	} else if (a.type == F32) {
		float* dst = a.data<float>();
		for (size_t i = 0; i < n; ++i)
			dst[i] = (float)fn((double)dst[i], x);
	} else {
		double* dst = a.data<double>();
		for (size_t i = 0; i < n; ++i)
			dst[i] = fn(dst[i], x);
	}
}

NDArray NDArray::atan2(const NDArray& x) const {
	requireSameShape(*this, x);
	NDArrayType rt = isFloatingPoint(type) && isFloatingPoint(x.type)
		? promoteTypes(type, x.type)
		: F64;
	NDArray y = convert(rt);
	NDArray xx = x.convert(rt);
	mapRealBinary(y, xx, std::atan2);
	return y;
}

NDArray NDArray::atan2(double x) const {
	NDArrayType rt = isFloatingPoint(type) ? type : typeForRealUnary(*this);
	NDArray y = convert(rt);
	mapRealBinaryScalar(y, x, std::atan2);
	return y;
}

NDArray NDArray::atan2(float x) const { return atan2((double)x); }
NDArray NDArray::atan2(int x) const { return atan2((double)x); }

NDArray NDArray::hypot(const NDArray& x) const {
	requireSameShape(*this, x);
	NDArrayType rt = isFloatingPoint(type) && isFloatingPoint(x.type)
		? promoteTypes(type, x.type)
		: F64;
	NDArray a = convert(rt);
	NDArray b = x.convert(rt);
	mapRealBinary(a, b, std::hypot);
	return a;
}

NDArray NDArray::hypot(double x) const {
	NDArrayType rt = isFloatingPoint(type) ? type : typeForRealUnary(*this);
	NDArray a = convert(rt);
	mapRealBinaryScalar(a, x, std::hypot);
	return a;
}

NDArray NDArray::hypot(float x) const { return hypot((double)x); }
NDArray NDArray::hypot(int x) const { return hypot((double)x); }


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
			case F16:
			case BF16:
				if (op == ElemBinOp::Min) {
					if (left.type == F16)
						ndhalf_min_f16(left.u16, left.u16, right.u16, n);
					else
						ndhalf_min_bf16(left.u16, left.u16, right.u16, n);
				} else if (op == ElemBinOp::Max) {
					if (left.type == F16)
						ndhalf_max_f16(left.u16, left.u16, right.u16, n);
					else
						ndhalf_max_bf16(left.u16, left.u16, right.u16, n);
				} else {
					for (size_t i = 0; i < n; ++i) {
						double av = ndarray_half::load(left.type, left.u16[i]);
						double bv = ndarray_half::load(right.type, right.u16[i]);
						left.u16[i] = ndarray_half::store(left.type, (float)elemBinDouble(av, bv, op));
					}
				}
				break;
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
			case INT8:
				for (size_t i = 0; i < n; ++i)
					left.int8[i] = (int8_t)elemBinI64(left.int8[i], right.int8[i], op);
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
			case INT3:
				for (size_t i = 0; i < n; ++i) {
					int av = int3_getSigned(left.uint64, i);
					int bv = int3_getSigned(right.uint64, i);
					int3_setSigned(left.uint64, i, (int)elemBinI64(av, bv, op));
				}
				int3_clearPadding(left.uint64, n);
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
		NDArrayType rt = promoteWithDoubleScalar(a.type, s);
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (left.type == F16 || left.type == BF16) {
			for (size_t i = 0; i < n; ++i) {
				float av = ndarray_half::load(left.type, left.u16[i]);
				left.u16[i] = ndarray_half::store(left.type, (float)elemBinDouble(av, s, op));
			}
		} else if (left.type == F64) {
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
		} else if (left.type == INT8) {
			int8_t sb = (int8_t)s;
			for (size_t i = 0; i < n; ++i)
				left.int8[i] = (int8_t)elemBinI64(left.int8[i], sb, op);
		} else if (left.type == UINT256) {
			uint256_t sb(s);
			for (size_t i = 0; i < n; ++i)
				left.uint256[i] = elemBinU256(left.uint256[i], sb, op);
		} else if (left.type == BINARY) {
			uint8_t sb = s > 0.0 ? 1 : 0;
			for (size_t i = 0; i < n; ++i) {
				uint8_t av = binaryGet(left.uint64, i);
				binarySet(left.uint64, i, elemBinU8(av, sb, op) & 1);
			}
		} else if (left.type == INT3) {
			int sb = (int)s;
			for (size_t i = 0; i < n; ++i) {
				int av = int3_getSigned(left.uint64, i);
				int3_setSigned(left.uint64, i, (int)elemBinI64(av, sb, op));
			}
			int3_clearPadding(left.uint64, n);
		} else {
			throw std::runtime_error("NDArray: invalid type in scalar element-wise op");
		}
		return left;
	}

	static NDArray elemBinFloatScalar(const NDArray& a, float s, ElemBinOp op) {
		// Prefer F32 when the array promotes there cleanly.
		NDArrayType rt = promoteWithFloatScalar(a.type, s);
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (left.type == F16 || left.type == BF16) {
			for (size_t i = 0; i < n; ++i) {
				float av = ndarray_half::load(left.type, left.u16[i]);
				left.u16[i] = ndarray_half::store(left.type, (float)elemBinDouble(av, s, op));
			}
		} else if (left.type == F32) {
			for (size_t i = 0; i < n; ++i)
				left.float32[i] = (float)elemBinDouble(left.float32[i], s, op);
		} else {
			// Fell into a wider type (e.g. F64) — use double path on converted array
			return elemBinDoubleScalar(a, (double)s, op);
		}
		return left;
	}

	static NDArray elemBinIntScalar(const NDArray& a, int s, ElemBinOp op) {
		NDArrayType rt = promoteWithIntScalar(a.type, s);
		NDArray left = a.convert(rt);
		const size_t n = left.numElements();
		if (isFloatingPoint(left.type))
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
		} else if (left.type == INT8) {
			int8_t sb = (int8_t)s;
			for (size_t i = 0; i < n; ++i)
				left.int8[i] = (int8_t)elemBinI64(left.int8[i], sb, op);
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
		} else if (left.type == INT3) {
			for (size_t i = 0; i < n; ++i) {
				int av = int3_getSigned(left.uint64, i);
				int3_setSigned(left.uint64, i, (int)elemBinI64(av, s, op));
			}
			int3_clearPadding(left.uint64, n);
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
				case F16:
				case BF16:
					r = cmpDouble(ndarray_half::load(ct, left.u16[i]),
					              ndarray_half::load(ct, right.u16[i]), op);
					break;
				case F32: r = cmpDouble(left.float32[i], right.float32[i], op); break;
				case F64: r = cmpDouble(left.float64[i], right.float64[i], op); break;
				case UINT8: r = cmpI64(left.uint8[i], right.uint8[i], op); break;
				case INT8: r = cmpI64(left.int8[i], right.int8[i], op); break;
				case INT3: r = cmpI64(int3_getSigned(left.uint64, i), int3_getSigned(right.uint64, i), op); break;
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
		if (a.type == F32)
			ct = F32;
		else if (a.type == F64 || isLosslessConversion(a.type, F64))
			ct = F64;
		else
			throw std::invalid_argument("NDArray: cannot compare this type with float scalar");

		NDArray left = a.convert(ct);
		NDArray out(a.shape, BINARY);
		const size_t n = left.numElements();
		for (size_t i = 0; i < n; ++i) {
			bool r = ct == F32
				? cmpDouble(left.float32[i], s, op)
				: cmpDouble(left.float64[i], s, op);
			binarySet(out.uint64, i, r ? 1 : 0);
		}
		return out;
	}

	/** Three-way: -1 if a<b, 0 if a==b, +1 if a>b. */
	static int8_t threeWayFromBools(bool lt, bool gt) {
		if (lt)
			return -1;
		if (gt)
			return 1;
		return 0;
	}

	static int8_t threeWayDouble(double a, double b) {
		return threeWayFromBools(a < b, a > b);
	}

	static int8_t threeWayI64(int64_t a, int64_t b) {
		return threeWayFromBools(a < b, a > b);
	}

	static int8_t threeWayU256(const uint256_t& a, const uint256_t& b) {
		return threeWayFromBools(a < b, b < a);
	}

	/**
	 * Element-wise three-way compare → INT3 (-1 / 0 / +1).
	 * Uses the same promotion rules as relational masks.
	 */
	static NDArray threeWayCompareArrays(const NDArray& a, const NDArray& b) {
		requireSameShape(a, b);
		NDArrayType ct = promoteTypes(a.type, b.type);
		if (a.type == BINARY && b.type == BINARY)
			ct = BINARY;
		if (a.type == INT3 && b.type == INT3)
			ct = INT3;

		NDArray left = a.convert(ct);
		NDArray right = b.convert(ct);
		NDArray out(a.shape, INT3);
		const size_t n = left.numElements();

		for (size_t i = 0; i < n; ++i) {
			int8_t r = 0;
			switch (ct) {
				case F16:
				case BF16:
					r = threeWayDouble(ndarray_half::load(ct, left.u16[i]),
					                   ndarray_half::load(ct, right.u16[i]));
					break;
				case F32: r = threeWayDouble(left.float32[i], right.float32[i]); break;
				case F64: r = threeWayDouble(left.float64[i], right.float64[i]); break;
				case UINT8: r = threeWayI64(left.uint8[i], right.uint8[i]); break;
				case INT8: r = threeWayI64(left.int8[i], right.int8[i]); break;
				case INT3: r = threeWayI64(int3_getSigned(left.uint64, i), int3_getSigned(right.uint64, i)); break;
				case INT32: r = threeWayI64(left.int32[i], right.int32[i]); break;
				case INT64: r = threeWayI64(left.int64[i], right.int64[i]); break;
				case UINT256: r = threeWayU256(left.uint256[i], right.uint256[i]); break;
				case BINARY: r = threeWayI64(binaryGet(left.uint64, i), binaryGet(right.uint64, i)); break;
				default: throw std::runtime_error("NDArray::compare: invalid type");
			}
			int3_setSigned(out.uint64, i, r);
		}
		return out;
	}

	static NDArray threeWayCompareDoubleScalar(const NDArray& a, double s) {
		// Compare in a type that can hold both; prefer native float width when possible.
		NDArray out(a.shape, INT3);
		const size_t n = a.numElements();
		if (a.type == F32) {
			for (size_t i = 0; i < n; ++i)
				int3_setSigned(out.uint64, i, threeWayDouble(a.float32[i], s));
			return out;
		}
		if (a.type == F64 || isFloatingPoint(a.type) || isLosslessConversion(a.type, F64)
		    || a.type == INT3 || a.type == INT8 || a.type == UINT8 || a.type == INT32 || a.type == INT64
		    || a.type == BINARY) {
			for (size_t i = 0; i < n; ++i)
				int3_setSigned(out.uint64, i, threeWayDouble(a.loadAsDouble(i), s));
			return out;
		}
		// UINT256 vs float: via double (lossy for huge ints; document)
		for (size_t i = 0; i < n; ++i)
			int3_setSigned(out.uint64, i, threeWayDouble(a.loadAsDouble(i), s));
		return out;
	}

	static NDArray threeWayCompareI64Scalar(const NDArray& a, int64_t s) {
		NDArray out(a.shape, INT3);
		const size_t n = a.numElements();
		if (a.type == UINT256) {
			uint256_t su = s < 0 ? uint256_t((int)s) : uint256_t((uint64_t)s);
			for (size_t i = 0; i < n; ++i)
				int3_setSigned(out.uint64, i, threeWayU256(a.uint256[i], su));
			return out;
		}
		if (isFloatingPoint(a.type)) {
			for (size_t i = 0; i < n; ++i)
				int3_setSigned(out.uint64, i, threeWayDouble(a.loadAsDouble(i), (double)s));
			return out;
		}
		for (size_t i = 0; i < n; ++i)
			int3_setSigned(out.uint64, i, threeWayI64(a.loadAsI64(i), s));
		return out;
	}

	static NDArray threeWayCompareU256Scalar(const NDArray& a, const uint256_t& s) {
		NDArray out(a.shape, INT3);
		const size_t n = a.numElements();
		if (a.type == UINT256) {
			for (size_t i = 0; i < n; ++i)
				int3_setSigned(out.uint64, i, threeWayU256(a.uint256[i], s));
			return out;
		}
		// Promote left to UINT256 for integer-like types; floats via double.
		if (isFloatingPoint(a.type)) {
			double sd = s.toDouble();
			for (size_t i = 0; i < n; ++i)
				int3_setSigned(out.uint64, i, threeWayDouble(a.loadAsDouble(i), sd));
			return out;
		}
		for (size_t i = 0; i < n; ++i) {
			uint256_t av = a.loadAsU256(i);
			int3_setSigned(out.uint64, i, threeWayU256(av, s));
		}
		return out;
	}

	static bool isZeroElement(const NDArray& a, size_t i) {
		switch (a.type) {
			case BINARY: return binaryGet(a.uint64, i) == 0;
			case INT3: return int3_get(a.uint64, i) == 0;
			case UINT8: return a.uint8[i] == 0;
			case INT8: return a.int8[i] == 0;
			case INT32: return a.int32[i] == 0;
			case INT64: return a.int64[i] == 0;
			case F32: return a.float32[i] == 0.0f;
			case F64: return a.float64[i] == 0.0;
			case UINT256: return a.uint256[i] == uint256_t(0);
			default: return a.loadAsDouble(i) == 0.0;
		}
	}

	static void copyElement(NDArray& out, size_t oi, const NDArray& src, size_t si) {
		switch (out.type) {
			case F16:
			case BF16: out.u16[oi] = src.u16[si]; break;
			case F32: out.float32[oi] = src.float32[si]; break;
			case F64: out.float64[oi] = src.float64[si]; break;
			case UINT8: out.uint8[oi] = src.uint8[si]; break;
			case INT8: out.int8[oi] = src.int8[si]; break;
			case INT3: int3_set(out.uint64, oi, int3_get(src.uint64, si)); break;
			case INT32: out.int32[oi] = src.int32[si]; break;
			case INT64: out.int64[oi] = src.int64[si]; break;
			case UINT256: out.uint256[oi] = src.uint256[si]; break;
			case BINARY:
				binarySet(out.uint64, oi, binaryGet(src.uint64, si));
				break;
			default: throw std::runtime_error("copyElement: invalid type");
		}
	}

	static void divElement(NDArray& out, size_t i, const NDArray& num, const NDArray& den) {
		switch (out.type) {
			case F16:
			case BF16:
				out.u16[i] = ndarray_half::store(out.type, arithF32(
					ndarray_half::load(num.type, num.u16[i]),
					ndarray_half::load(den.type, den.u16[i]),
					ArithOp::Div));
				break;
			case F32: out.float32[i] = num.float32[i] / den.float32[i]; break;
			case F64: out.float64[i] = num.float64[i] / den.float64[i]; break;
			case UINT8: out.uint8[i] = (uint8_t)(num.uint8[i] / den.uint8[i]); break;
			case INT8: out.int8[i] = (int8_t)(num.int8[i] / den.int8[i]); break;
			case INT3: {
				int a = int3_getSigned(num.uint64, i);
				int b = int3_getSigned(den.uint64, i);
				int3_setSigned(out.uint64, i, a / b);
				break;
			}
			case INT32: out.int32[i] = num.int32[i] / den.int32[i]; break;
			case INT64: out.int64[i] = num.int64[i] / den.int64[i]; break;
			case UINT256: out.uint256[i] = num.uint256[i] / den.uint256[i]; break;
			case BINARY:
				binarySet(out.uint64, i, binaryGet(num.uint64, i));
				break;
			default: throw std::runtime_error("divElement: invalid type");
		}
	}

	// ---- ArrayOrScalar helpers (shared by select / safeDiv / piecewise) ----

	static NDArrayType aosType(const ArrayOrScalar& p) {
		if (p.kind == ArrayOrScalar::Array)
			return p.arr->type;
		if (p.kind == ArrayOrScalar::ScalarDouble)
			return F64;
		return INT32;
	}

	static NDArrayType promoteAos(const ArrayOrScalar* args, int nArgs) {
		bool haveArr = false;
		NDArrayType rt = F64;
		for (int k = 0; k < nArgs; ++k) {
			if (args[k].kind == ArrayOrScalar::Array) {
				if (!haveArr) {
					rt = args[k].arr->type;
					haveArr = true;
				} else {
					rt = promoteTypes(rt, args[k].arr->type);
				}
			}
		}
		if (!haveArr) {
			bool anyDouble = false;
			bool allTernaryInts = true;
			bool anyInt = false;
			for (int k = 0; k < nArgs; ++k) {
				if (args[k].kind == ArrayOrScalar::ScalarDouble)
					anyDouble = true;
				else if (args[k].kind == ArrayOrScalar::ScalarInt) {
					anyInt = true;
					int64_t v = args[k].i;
					if (v != -1 && v != 0 && v != 1)
						allTernaryInts = false;
				}
			}
			if (anyDouble)
				return F64;
			// Pure int scalars in {-1,0,1} → INT3 (matches NDArray(int) / full defaults)
			if (anyInt && allTernaryInts)
				return INT3;
			return INT32;
		}
		for (int k = 0; k < nArgs; ++k) {
			if (args[k].kind == ArrayOrScalar::ScalarDouble)
				rt = promoteTypes(rt, F64);
			else if (args[k].kind == ArrayOrScalar::ScalarInt)
				// -1/0/1 promote as INT3 so where(mask, -1, int3Arr) stays INT3
				rt = promoteTypes(rt, typeForIntConstant(args[k].i));
		}
		return rt;
	}

	/** Also promote against a fixed array type (e.g. num/den for safeDiv). */
	static NDArrayType promoteAosWith(NDArrayType base, const ArrayOrScalar& p) {
		ArrayOrScalar tmp[1] = { p };
		NDArrayType t = promoteAos(tmp, 1);
		return promoteTypes(base, t);
	}

	static const NDArray* prepAosArray(const ArrayOrScalar& p, NDArrayType rt, std::unique_ptr<NDArray>& storage) {
		if (p.kind != ArrayOrScalar::Array)
			return nullptr;
		if (p.arr->type == rt)
			return p.arr;
		storage = std::make_unique<NDArray>(p.arr->convert(rt));
		return storage.get();
	}

	static void writeAos(NDArray& out, size_t i, const ArrayOrScalar& p, const NDArray* arrConv) {
		if (p.kind == ArrayOrScalar::Array)
			copyElement(out, i, *arrConv, i);
		else if (p.kind == ArrayOrScalar::ScalarDouble)
			out.storeFromDouble(i, p.d);
		else
			out.storeFromI64(i, p.i);
	}

	static void requireAosShape(const NDArray& m, const ArrayOrScalar& v) {
		if (v.kind == ArrayOrScalar::Array)
			requireSameShape(m, *v.arr);
	}

	/** Single-pass select: out[i] = truthy(cond[i]) ? x : y (array or scalar). */
	static NDArray select(const NDArray& condition, ArrayOrScalar x, ArrayOrScalar y) {
		requireAosShape(condition, x);
		requireAosShape(condition, y);
		ArrayOrScalar args[2] = { x, y };
		NDArrayType rt = promoteAos(args, 2);
		std::unique_ptr<NDArray> storX, storY;
		const NDArray* ax = prepAosArray(x, rt, storX);
		const NDArray* ay = prepAosArray(y, rt, storY);
		NDArray out(condition.shape, rt);
		const size_t n = condition.numElements();
		for (size_t i = 0; i < n; ++i) {
			if (condition.loadAsDouble(i) != 0.0)
				writeAos(out, i, x, ax);
			else
				writeAos(out, i, y, ay);
		}
		return out;
	}

	/** num / den with zero-den → whenZero (scalar or array). */
	static NDArray safeDiv(const NDArray& num, const NDArray& den, const ArrayOrScalar& whenZero) {
		requireSameShape(num, den);
		requireAosShape(num, whenZero);
		NDArrayType rt = promoteTypes(num.type, den.type);
		rt = promoteAosWith(rt, whenZero);
		NDArray N = num.convert(rt);
		NDArray D = den.convert(rt);
		std::unique_ptr<NDArray> storZ;
		const NDArray* aZ = prepAosArray(whenZero, rt, storZ);
		NDArray out(num.shape, rt);
		const size_t n = num.numElements();
		for (size_t i = 0; i < n; ++i) {
			if (isZeroElement(D, i))
				writeAos(out, i, whenZero, aZ);
			else
				divElement(out, i, N, D);
		}
		return out;
	}

	static NDArray piecewise1(const NDArray& m0, const ArrayOrScalar& v0, const ArrayOrScalar& otherwise) {
		requireAosShape(m0, v0);
		requireAosShape(m0, otherwise);
		ArrayOrScalar args[2] = { v0, otherwise };
		NDArrayType rt = promoteAos(args, 2);
		std::unique_ptr<NDArray> stor0, storO;
		const NDArray* a0 = prepAosArray(v0, rt, stor0);
		const NDArray* aO = prepAosArray(otherwise, rt, storO);
		NDArray out(m0.shape, rt);
		const size_t n = m0.numElements();
		for (size_t i = 0; i < n; ++i) {
			if (m0.loadAsDouble(i) != 0.0)
				writeAos(out, i, v0, a0);
			else
				writeAos(out, i, otherwise, aO);
		}
		return out;
	}

	static NDArray piecewise2(const NDArray& m0, ArrayOrScalar v0,
	                          const NDArray& m1, ArrayOrScalar v1,
	                          ArrayOrScalar otherwise) {
		requireSameShape(m0, m1);
		requireAosShape(m0, v0);
		requireAosShape(m0, v1);
		requireAosShape(m0, otherwise);
		ArrayOrScalar args[3] = { v0, v1, otherwise };
		NDArrayType rt = promoteAos(args, 3);
		std::unique_ptr<NDArray> stor0, stor1, storO;
		const NDArray* a0 = prepAosArray(v0, rt, stor0);
		const NDArray* a1 = prepAosArray(v1, rt, stor1);
		const NDArray* aO = prepAosArray(otherwise, rt, storO);
		NDArray out(m0.shape, rt);
		const size_t n = m0.numElements();
		for (size_t i = 0; i < n; ++i) {
			if (m0.loadAsDouble(i) != 0.0)
				writeAos(out, i, v0, a0);
			else if (m1.loadAsDouble(i) != 0.0)
				writeAos(out, i, v1, a1);
			else
				writeAos(out, i, otherwise, aO);
		}
		return out;
	}

	static NDArray piecewise3(const NDArray& m0, ArrayOrScalar v0,
	                          const NDArray& m1, ArrayOrScalar v1,
	                          const NDArray& m2, ArrayOrScalar v2,
	                          ArrayOrScalar otherwise) {
		requireSameShape(m0, m1);
		requireSameShape(m0, m2);
		requireAosShape(m0, v0);
		requireAosShape(m0, v1);
		requireAosShape(m0, v2);
		requireAosShape(m0, otherwise);
		ArrayOrScalar args[4] = { v0, v1, v2, otherwise };
		NDArrayType rt = promoteAos(args, 4);
		std::unique_ptr<NDArray> stor0, stor1, stor2, storO;
		const NDArray* a0 = prepAosArray(v0, rt, stor0);
		const NDArray* a1 = prepAosArray(v1, rt, stor1);
		const NDArray* a2 = prepAosArray(v2, rt, stor2);
		const NDArray* aO = prepAosArray(otherwise, rt, storO);
		NDArray out(m0.shape, rt);
		const size_t n = m0.numElements();
		for (size_t i = 0; i < n; ++i) {
			if (m0.loadAsDouble(i) != 0.0)
				writeAos(out, i, v0, a0);
			else if (m1.loadAsDouble(i) != 0.0)
				writeAos(out, i, v1, a1);
			else if (m2.loadAsDouble(i) != 0.0)
				writeAos(out, i, v2, a2);
			else
				writeAos(out, i, otherwise, aO);
		}
		return out;
	}

	// ---- reductions (abbreviated helpers use load/store) -------------------

	static NDArray reduceAll(const NDArray& a, ReduceOp op);
	static NDArray reduceAxis(const NDArray& a, int axis, ReduceOp op);
};

// Define reduceAll / reduceAxis outside struct for size (still as Impl members)

template <typename Acc>
static Acc laneToAcc(const NDArray& a, size_t i) {
	switch (a.type) {
		case F16:
		case BF16:    return Acc(ndarray_half::load(a.type, a.data<uint16_t>()[i]));
		case F32:     return accFromLane<float, Acc>(a.data<float>()[i]);
		case F64:     return accFromLane<double, Acc>(a.data<double>()[i]);
		case UINT8:   return accFromLane<uint8_t, Acc>(a.data<uint8_t>()[i]);
		case INT8:    return accFromLane<int8_t, Acc>(a.data<int8_t>()[i]);
		case INT32:   return accFromLane<int32_t, Acc>(a.data<int32_t>()[i]);
		case INT64:   return accFromLane<int64_t, Acc>(a.data<int64_t>()[i]);
		case INT3:    return accFromSigned<Acc>(int3_getSigned(a.data<uint64_t>(), i));
		case BINARY:  return accFromLane<uint8_t, Acc>(binaryGet(a.data<uint64_t>(), i));
		case UINT256: return accFromLane<uint256_t, Acc>(a.data<uint256_t>()[i]);
		default:
			throw std::runtime_error("NDArray: invalid type in reduction");
	}
}

template <typename Acc>
static Acc sumRange(const NDArray& a, size_t first, size_t count, size_t stride) {
	if (stride == 1) {
		switch (a.type) {
			case F16:
				if constexpr (std::is_same_v<Acc, float>)
					return ndhalf_sum_f16(a.data<uint16_t>() + first, count);
				return Acc(ndhalf_sum_f16(a.data<uint16_t>() + first, count));
			case BF16:
				if constexpr (std::is_same_v<Acc, float>)
					return ndhalf_sum_bf16(a.data<uint16_t>() + first, count);
				return Acc(ndhalf_sum_bf16(a.data<uint16_t>() + first, count));
			case F32:
				if constexpr (std::is_same_v<Acc, float>)
					return ndreduce_sum_f32(a.data<float>() + first, count);
				return denseSum<float, Acc>(a.data<float>() + first, count);
			case F64:
				return denseSum<double, Acc>(a.data<double>() + first, count);
			case UINT8:
				return denseSum<uint8_t, Acc>(a.data<uint8_t>() + first, count);
			case INT8:
				return denseSum<int8_t, Acc>(a.data<int8_t>() + first, count);
			case INT32:
				if constexpr (std::is_same_v<Acc, int32_t>)
					return ndreduce_sum_i32(a.data<int32_t>() + first, count);
				return denseSum<int32_t, Acc>(a.data<int32_t>() + first, count);
			case INT64:
				return denseSum<int64_t, Acc>(a.data<int64_t>() + first, count);
			case INT3: {
				const uint64_t* w = a.data<uint64_t>();
				Acc s{};
				for (size_t i = 0; i < count; ++i)
					s += accFromSigned<Acc>(int3_getSigned(w, first + i));
				return s;
			}
			case BINARY:
				return accFromKernelI64<Acc>(ndscore_binary_i64(
					a.data<uint64_t>(), first, a.data<uint64_t>(), first, count, NDScoreOp::L2Self));
			case UINT256:
				if constexpr (std::is_same_v<Acc, uint256_t>)
					return denseSum<uint256_t, Acc>(a.data<uint256_t>() + first, count);
				else {
					Acc s{};
					const uint256_t* p = a.data<uint256_t>() + first;
					for (size_t i = 0; i < count; ++i)
						s += accFromDouble<Acc>(p[i].toDouble());
					return s;
				}
			default:
				break;
		}
	}
	Acc s{};
	for (size_t r = 0; r < count; ++r)
		s += laneToAcc<Acc>(a, first + r * stride);
	return s;
}

template <typename Acc>
static Acc prodRange(const NDArray& a, size_t first, size_t count, size_t stride) {
	if (stride == 1) {
		switch (a.type) {
			case F16:   return Acc(ndhalf_prod_f16(a.data<uint16_t>() + first, count));
			case BF16:  return Acc(ndhalf_prod_bf16(a.data<uint16_t>() + first, count));
			case F32:   return denseProd<float, Acc>(a.data<float>() + first, count);
			case F64:   return denseProd<double, Acc>(a.data<double>() + first, count);
			case UINT8: return denseProd<uint8_t, Acc>(a.data<uint8_t>() + first, count);
			case INT8:  return denseProd<int8_t, Acc>(a.data<int8_t>() + first, count);
			case INT32: return denseProd<int32_t, Acc>(a.data<int32_t>() + first, count);
			case INT64: return denseProd<int64_t, Acc>(a.data<int64_t>() + first, count);
			case UINT256:
				if constexpr (std::is_same_v<Acc, uint256_t>)
					return denseProd<uint256_t, Acc>(a.data<uint256_t>() + first, count);
				else {
					Acc p = Acc(1);
					const uint256_t* ptr = a.data<uint256_t>() + first;
					for (size_t i = 0; i < count; ++i)
						accMulEq(p, accFromDouble<Acc>(ptr[i].toDouble()));
					return p;
				}
			default: break;
		}
	}
	Acc p = Acc(1);
	for (size_t r = 0; r < count; ++r)
		accMulEq(p, laneToAcc<Acc>(a, first + r * stride));
	return p;
}

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
			case F16:
			case BF16: {
				float acc = ndarray_half::load(a.type, a.u16[0]);
				for (size_t i = 1; i < n; ++i) {
					float v = ndarray_half::load(a.type, a.u16[i]);
					acc = op == ReduceOp::Min ? std::fmin(acc, v) : std::fmax(acc, v);
				}
				out.u16[0] = ndarray_half::store(a.type, acc);
				break;
			}
			case F32: {
				float acc = a.float32[0];
				for (size_t i = 1; i < n; ++i)
					acc = op == ReduceOp::Min ? std::fmin(acc, a.float32[i]) : std::fmax(acc, a.float32[i]);
				out.float32[0] = acc;
				break;
			}
			case F64: {
				double acc = a.float64[0];
				for (size_t i = 1; i < n; ++i)
					acc = op == ReduceOp::Min ? std::fmin(acc, a.float64[i]) : std::fmax(acc, a.float64[i]);
				out.float64[0] = acc;
				break;
			}
			case UINT8: {
				uint8_t acc = a.uint8[0];
				for (size_t i = 1; i < n; ++i) {
					uint8_t v = a.uint8[i];
					acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
				}
				out.uint8[0] = acc;
				break;
			}
			case INT8: {
				int8_t acc = a.int8[0];
				for (size_t i = 1; i < n; ++i) {
					int8_t v = a.int8[i];
					acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
				}
				out.int8[0] = acc;
				break;
			}
			case INT3: {
				int acc = int3_getSigned(a.uint64, 0);
				for (size_t i = 1; i < n; ++i) {
					int v = int3_getSigned(a.uint64, i);
					acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
				}
				int3_setSigned(out.uint64, 0, acc);
				break;
			}
			case INT32: {
				int32_t acc = a.int32[0];
				for (size_t i = 1; i < n; ++i) {
					int32_t v = a.int32[i];
					acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
				}
				out.int32[0] = acc;
				break;
			}
			case INT64: {
				int64_t acc = a.int64[0];
				for (size_t i = 1; i < n; ++i) {
					int64_t v = a.int64[i];
					acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
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
					acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
				}
				binarySet(out.uint64, 0, acc);
				break;
			}
			default: throw std::runtime_error("min/max reduce: invalid type");
		}
		return out;
	}

	// Sum / Prod — Acc kernels (sumAs / prodAs), lossless default type.
	NDArrayType rt = sumProdAccumulateType(a.type);
	NDArray out({}, rt);
	if (op == ReduceOp::Sum) {
		switch (rt) {
			case F32:     out.float32[0] = a.sumAs<float>(); break;
			case F64:     out.float64[0] = a.sumAs<double>(); break;
			case INT32:   out.int32[0] = a.sumAs<int32_t>(); break;
			case INT64:   out.int64[0] = a.sumAs<int64_t>(); break;
			default:      out.uint256[0] = a.sumAs<uint256_t>(); break;
		}
	} else {
		switch (rt) {
			case F32:     out.float32[0] = a.prodAs<float>(); break;
			case F64:     out.float64[0] = a.prodAs<double>(); break;
			case INT32:   out.int32[0] = a.prodAs<int32_t>(); break;
			case INT64:   out.int64[0] = a.prodAs<int64_t>(); break;
			default:      out.uint256[0] = a.prodAs<uint256_t>(); break;
		}
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
				case F16:
				case BF16: {
					float acc = ndarray_half::load(a.type, a.u16[first]);
					for (size_t r = 1; r < reduced; ++r) {
						float v = ndarray_half::load(a.type, a.u16[flatIndexWithAxis(a.shape, axis, oi, r)]);
						acc = op == ReduceOp::Min ? std::fmin(acc, v) : std::fmax(acc, v);
					}
					out.u16[oi] = ndarray_half::store(a.type, acc);
					break;
				}
				case F32: {
					float acc = a.float32[first];
					for (size_t r = 1; r < reduced; ++r) {
						float v = a.float32[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = op == ReduceOp::Min ? std::fmin(acc, v) : std::fmax(acc, v);
					}
					out.float32[oi] = acc;
					break;
				}
				case F64: {
					double acc = a.float64[first];
					for (size_t r = 1; r < reduced; ++r) {
						double v = a.float64[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = op == ReduceOp::Min ? std::fmin(acc, v) : std::fmax(acc, v);
					}
					out.float64[oi] = acc;
					break;
				}
				case UINT8: {
					uint8_t acc = a.uint8[first];
					for (size_t r = 1; r < reduced; ++r) {
						uint8_t v = a.uint8[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
					}
					out.uint8[oi] = acc;
					break;
				}
				case INT8: {
					int8_t acc = a.int8[first];
					for (size_t r = 1; r < reduced; ++r) {
						int8_t v = a.int8[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
					}
					out.int8[oi] = acc;
					break;
				}
				case INT3: {
					int acc = int3_getSigned(a.uint64, first);
					for (size_t r = 1; r < reduced; ++r) {
						int v = int3_getSigned(a.uint64, flatIndexWithAxis(a.shape, axis, oi, r));
						acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
					}
					int3_setSigned(out.uint64, oi, acc);
					break;
				}
				case INT32: {
					int32_t acc = a.int32[first];
					for (size_t r = 1; r < reduced; ++r) {
						int32_t v = a.int32[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
					}
					out.int32[oi] = acc;
					break;
				}
				case INT64: {
					int64_t acc = a.int64[first];
					for (size_t r = 1; r < reduced; ++r) {
						int64_t v = a.int64[flatIndexWithAxis(a.shape, axis, oi, r)];
						acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
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
						acc = op == ReduceOp::Min ? (v < acc ? v : acc) : v > acc ? v : acc;
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
	size_t stride = 1;
	for (int d = axis + 1; d < a.shape.size(); ++d)
		stride *= (size_t)a.shape.get(d);
	for (size_t oi = 0; oi < outN; ++oi) {
		const size_t first = flatIndexWithAxis(a.shape, axis, oi, 0);
		if (op == ReduceOp::Sum) {
			switch (rt) {
				case F32:   out.float32[oi] = sumRange<float>(a, first, reduced, stride); break;
				case F64:   out.float64[oi] = sumRange<double>(a, first, reduced, stride); break;
				case INT32: out.int32[oi] = sumRange<int32_t>(a, first, reduced, stride); break;
				case INT64: out.int64[oi] = sumRange<int64_t>(a, first, reduced, stride); break;
				default:    out.uint256[oi] = sumRange<uint256_t>(a, first, reduced, stride); break;
			}
		} else {
			switch (rt) {
				case F32:   out.float32[oi] = prodRange<float>(a, first, reduced, stride); break;
				case F64:   out.float64[oi] = prodRange<double>(a, first, reduced, stride); break;
				case INT32: out.int32[oi] = prodRange<int32_t>(a, first, reduced, stride); break;
				case INT64: out.int64[oi] = prodRange<int64_t>(a, first, reduced, stride); break;
				default:    out.uint256[oi] = prodRange<uint256_t>(a, first, reduced, stride); break;
			}
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

// int64 scalar min/max/pow/mod: promote to INT64 then run integer kernel
NDArray NDArray::minimum(int64_t other) const {
	NDArrayType rt = promoteWithInt64Scalar(type, other);
	if (isFloatingPoint(rt))
		return minimum((double)other);
	NDArray left = convert(rt);
	const size_t n = left.numElements();
	if (left.type == INT64) {
		for (size_t i = 0; i < n; ++i)
			left.int64[i] = left.int64[i] < other ? left.int64[i] : other;
	} else if (left.type == INT32) {
		int32_t s = (int32_t)other;
		for (size_t i = 0; i < n; ++i)
			left.int32[i] = left.int32[i] < s ? left.int32[i] : s;
	} else {
		return Impl::elemBinIntScalar(*this, (int)other, ElemBinOp::Min);
	}
	return left;
}
NDArray NDArray::maximum(int64_t other) const {
	NDArrayType rt = promoteWithInt64Scalar(type, other);
	if (isFloatingPoint(rt))
		return maximum((double)other);
	NDArray left = convert(rt);
	const size_t n = left.numElements();
	if (left.type == INT64) {
		for (size_t i = 0; i < n; ++i)
			left.int64[i] = left.int64[i] > other ? left.int64[i] : other;
	} else if (left.type == INT32) {
		int32_t s = (int32_t)other;
		for (size_t i = 0; i < n; ++i)
			left.int32[i] = left.int32[i] > s ? left.int32[i] : s;
	} else {
		return Impl::elemBinIntScalar(*this, (int)other, ElemBinOp::Max);
	}
	return left;
}
NDArray NDArray::pow(int64_t other) const {
	NDArrayType rt = promoteWithInt64Scalar(type, other);
	if (isFloatingPoint(rt))
		return pow((double)other);
	NDArray left = convert(rt);
	const size_t n = left.numElements();
	for (size_t i = 0; i < n; ++i) {
		int64_t base = left.loadAsI64(i);
		int64_t r = 1;
		int64_t b = base;
		uint64_t exp = other < 0 ? 0 : (uint64_t)other;
		if (other < 0)
			throw std::invalid_argument("NDArray::pow: negative exponent on integer");
		while (exp) {
			if (exp & 1ULL) r *= b;
			b *= b;
			exp >>= 1;
		}
		left.storeFromI64(i, r);
	}
	return left;
}
NDArray NDArray::mod(int64_t other) const {
	NDArrayType rt = promoteWithInt64Scalar(type, other);
	if (isFloatingPoint(rt))
		return mod((double)other);
	if (other == 0)
		throw std::invalid_argument("NDArray::mod: division by zero");
	NDArray left = convert(rt);
	const size_t n = left.numElements();
	for (size_t i = 0; i < n; ++i)
		left.storeFromI64(i, left.loadAsI64(i) % other);
	return left;
}

NDArray NDArray::compare(const NDArray& other) const {
	return Impl::threeWayCompareArrays(*this, other);
}
NDArray NDArray::compare(int other) const {
	return Impl::threeWayCompareI64Scalar(*this, (int64_t)other);
}
NDArray NDArray::compare(float other) const {
	return Impl::threeWayCompareDoubleScalar(*this, (double)other);
}
NDArray NDArray::compare(double other) const {
	return Impl::threeWayCompareDoubleScalar(*this, other);
}
NDArray NDArray::compare(const uint256_t& other) const {
	return Impl::threeWayCompareU256Scalar(*this, other);
}

NDArray NDArray::equal(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Eq); }
NDArray NDArray::notEqual(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Ne); }
NDArray NDArray::less(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Lt); }
NDArray NDArray::lessEqual(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Le); }
NDArray NDArray::greater(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Gt); }
NDArray NDArray::greaterEqual(const NDArray& other) const { return Impl::compareArrays(*this, other, CmpOp::Ge); }

// Element-wise relational operators (BINARY masks).
// operator==(const NDArray&) remains whole-array bool (defined above).
// operator==(scalar) is element-wise BINARY.
NDArray NDArray::operator>(const NDArray& other) const { return greater(other); }
NDArray NDArray::operator<(const NDArray& other) const { return less(other); }
NDArray NDArray::operator>=(const NDArray& other) const { return greaterEqual(other); }
NDArray NDArray::operator<=(const NDArray& other) const { return lessEqual(other); }
NDArray NDArray::operator!=(const NDArray& other) const { return notEqual(other); }

NDArray NDArray::operator==(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Eq); }
NDArray NDArray::operator!=(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ne); }
NDArray NDArray::operator>(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Gt); }
NDArray NDArray::operator<(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Lt); }
NDArray NDArray::operator>=(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ge); }
NDArray NDArray::operator<=(float other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Le); }

NDArray NDArray::operator==(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Eq); }
NDArray NDArray::operator!=(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ne); }
NDArray NDArray::operator>(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Gt); }
NDArray NDArray::operator<(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Lt); }
NDArray NDArray::operator>=(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Ge); }
NDArray NDArray::operator<=(double other) const { return Impl::compareDoubleScalar(*this, other, CmpOp::Le); }

NDArray NDArray::operator==(int other) const { return Impl::compareDoubleScalar(*this, (double)other, CmpOp::Eq); }
NDArray NDArray::operator!=(int other) const { return Impl::compareDoubleScalar(*this, (double)other, CmpOp::Ne); }
NDArray NDArray::operator>(int other) const { return Impl::compareDoubleScalar(*this, (double)other, CmpOp::Gt); }
NDArray NDArray::operator<(int other) const { return Impl::compareDoubleScalar(*this, (double)other, CmpOp::Lt); }
NDArray NDArray::operator>=(int other) const { return Impl::compareDoubleScalar(*this, (double)other, CmpOp::Ge); }
NDArray NDArray::operator<=(int other) const { return Impl::compareDoubleScalar(*this, (double)other, CmpOp::Le); }

// ArrayOrScalar
NDArray::ArrayOrScalar::ArrayOrScalar(const NDArray& a) : kind(Array), arr(&a), d(0), i(0) {}
NDArray::ArrayOrScalar::ArrayOrScalar(double v) : kind(ScalarDouble), arr(nullptr), d(v), i(0) {}
NDArray::ArrayOrScalar::ArrayOrScalar(float v) : kind(ScalarDouble), arr(nullptr), d((double)v), i(0) {}
NDArray::ArrayOrScalar::ArrayOrScalar(int v) : kind(ScalarInt), arr(nullptr), d(0), i((int64_t)v) {}
NDArray::ArrayOrScalar::ArrayOrScalar(int64_t v) : kind(ScalarInt), arr(nullptr), d(0), i(v) {}

NDArray NDArray::where(const NDArray& condition, ArrayOrScalar x, ArrayOrScalar y) {
	return Impl::select(condition, x, y);
}

NDArray NDArray::select(const NDArray& condition, ArrayOrScalar x, ArrayOrScalar y) {
	return Impl::select(condition, x, y);
}

NDArray NDArray::choose(ArrayOrScalar ifTrue, ArrayOrScalar ifFalse) const {
	return Impl::select(*this, ifTrue, ifFalse);
}

NDArray NDArray::safeDiv(const NDArray& den, ArrayOrScalar whenZero) const {
	return Impl::safeDiv(*this, den, whenZero);
}

NDArray NDArray::piecewise(const NDArray& m0, ArrayOrScalar v0, ArrayOrScalar otherwise) {
	return Impl::piecewise1(m0, v0, otherwise);
}

NDArray NDArray::piecewise(const NDArray& m0, ArrayOrScalar v0,
                           const NDArray& m1, ArrayOrScalar v1, ArrayOrScalar otherwise) {
	return Impl::piecewise2(m0, v0, m1, v1, otherwise);
}

NDArray NDArray::piecewise(const NDArray& m0, ArrayOrScalar v0,
                           const NDArray& m1, ArrayOrScalar v1,
                           const NDArray& m2, ArrayOrScalar v2, ArrayOrScalar otherwise) {
	return Impl::piecewise3(m0, v0, m1, v1, m2, v2, otherwise);
}

// ---- BINARY bulk fill + float→BINARY classification kernels (file-static) -----

/** Set the first nElems bits of a BINARY buffer to 1 (rest of last word 0). */
static void binaryFillOnes(uint64_t* words, size_t nElems) {
	const size_t fullWords = nElems / 64;
	for (size_t w = 0; w < fullWords; ++w)
		words[w] = ~uint64_t(0);
	const unsigned rem = (unsigned)(nElems % 64);
	if (rem)
		words[fullWords] = (1ULL << rem) - 1ULL;
}

/** Invert first nElems bits in place; clear unused high bits of last word. */
static void binaryInvertInPlace(uint64_t* words, size_t nElems) {
	const size_t fullWords = nElems / 64;
	for (size_t w = 0; w < fullWords; ++w)
		words[w] = ~words[w];
	const unsigned rem = (unsigned)(nElems % 64);
	if (rem)
		words[fullWords] = ~words[fullWords] & ((1ULL << rem) - 1ULL);
}

/** Non-zero test → BINARY, word-packed (F32). */
static void nonzeroF32ToBinary(uint64_t* outWords, const float* src, size_t n) {
	size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
	const __m512 z = _mm512_setzero_ps();
	for (; i + 64 <= n; i += 64) {
		uint64_t word = 0;
		for (int chunk = 0; chunk < 4; ++chunk) {
			__m512 v = _mm512_loadu_ps(src + i + chunk * 16);
			// NEQ_UQ: true for NaN as well (matches C++ bool(nan))
			__mmask16 m = _mm512_cmp_ps_mask(v, z, _CMP_NEQ_UQ);
			word |= (uint64_t)m << (chunk * 16);
		}
		outWords[i / 64] = word;
	}
#endif
	for (; i < n; ) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			// != 0 is true for NaN
			if (src[base + j] != 0.0f)
				word |= uint64_t(1) << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}

static void nonzeroF64ToBinary(uint64_t* outWords, const double* src, size_t n) {
	size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
	const __m512d z = _mm512_setzero_pd();
	for (; i + 64 <= n; i += 64) {
		uint64_t word = 0;
		for (int chunk = 0; chunk < 8; ++chunk) {
			__m512d v = _mm512_loadu_pd(src + i + chunk * 8);
			__mmask8 m = _mm512_cmp_pd_mask(v, z, _CMP_NEQ_UQ);
			word |= (uint64_t)m << (chunk * 8);
		}
		outWords[i / 64] = word;
	}
#endif
	for (; i < n; ) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			if (src[base + j] != 0.0)
				word |= uint64_t(1) << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}

template <typename T>
static void nonzeroIntegralToBinary(uint64_t* outWords, const T* src, size_t n) {
	size_t i = 0;
	for (; i < n; ) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			if (src[base + j] != T(0))
				word |= uint64_t(1) << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}

static void nonzeroInt3ToBinary(uint64_t* outWords, const uint64_t* srcWords, size_t n) {
	for (size_t i = 0; i < n; ) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			if (int3_get(srcWords, base + j) != 0)
				word |= uint64_t(1) << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}

static void nonzeroU256ToBinary(uint64_t* outWords, const uint256_t* src, size_t n) {
	const uint256_t z(0);
	for (size_t i = 0; i < n; ) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			if (!(src[base + j] == z))
				word |= uint64_t(1) << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}

// IEEE (abs bits): finite ⇔ abs < infBits; inf ⇔ abs == infBits; NaN ⇔ abs > infBits
enum class FloatClass { Finite, Infinite, NaN };

static inline bool f32_match(uint32_t u, FloatClass cls) {
	const uint32_t absv = u & 0x7fffffffu;
	switch (cls) {
		case FloatClass::Finite: return absv < 0x7f800000u;
		case FloatClass::Infinite: return absv == 0x7f800000u;
		case FloatClass::NaN: return absv > 0x7f800000u;
	}
	return false;
}

static inline bool f64_match(uint64_t u, FloatClass cls) {
	const uint64_t absv = u & 0x7fffffffffffffffull;
	switch (cls) {
		case FloatClass::Finite: return absv < 0x7ff0000000000000ull;
		case FloatClass::Infinite: return absv == 0x7ff0000000000000ull;
		case FloatClass::NaN: return absv > 0x7ff0000000000000ull;
	}
	return false;
}

// F16 inf/nan threshold 0x7c00; BF16 uses F32's top 16 bits (0x7f80).
static inline bool half_match(uint16_t h, NDArrayType t, FloatClass cls) {
	const uint16_t absv = (uint16_t)(h & 0x7fffu);
	const uint16_t infb = t == F16 ? (uint16_t)0x7c00u : (uint16_t)0x7f80u;
	switch (cls) {
		case FloatClass::Finite: return absv < infb;
		case FloatClass::Infinite: return absv == infb;
		case FloatClass::NaN: return absv > infb;
	}
	return false;
}

static void classifyHalfToBinary(uint64_t* outWords, const uint16_t* src, size_t n,
                                 NDArrayType t, FloatClass cls) {
	size_t i = 0;
	while (i < n) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			if (half_match(src[base + j], t, cls))
				word |= (uint64_t)1 << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}

/** Classify F32 → BINARY: one output word at a time (64 lanes). Portable path. */
static void classifyF32ToBinary(uint64_t* outWords, const float* src, size_t n, FloatClass cls) {
	size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
	// 64 floats = 4× ZMM → one BINARY word
	const __m512i expMask = _mm512_set1_epi32((int)0x7fffffff);
	const __m512i infBits = _mm512_set1_epi32((int)0x7f800000);
	for (; i + 64 <= n; i += 64) {
		uint64_t word = 0;
		for (int chunk = 0; chunk < 4; ++chunk) {
			__m512 v = _mm512_loadu_ps(src + i + chunk * 16);
			__m512i bits = _mm512_castps_si512(v);
			__m512i absv = _mm512_and_si512(bits, expMask);
			__mmask16 m;
			switch (cls) {
				case FloatClass::Finite:
					m = _mm512_cmplt_epu32_mask(absv, infBits);
					break;
				case FloatClass::Infinite:
					m = _mm512_cmpeq_epi32_mask(absv, infBits);
					break;
				case FloatClass::NaN:
					m = _mm512_cmpgt_epu32_mask(absv, infBits);
					break;
			}
			word |= (uint64_t)m << (chunk * 16);
		}
		outWords[i / 64] = word;
	}
#endif
	// Remainder / portable: pack up to 64 bits per word without per-bit RMW
	for (; i < n; ) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			uint32_t u;
			std::memcpy(&u, src + base + j, sizeof(u));
			if (f32_match(u, cls))
				word |= uint64_t(1) << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}

static void classifyF64ToBinary(uint64_t* outWords, const double* src, size_t n, FloatClass cls) {
	size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
	// 64 doubles = 8× ZMM → one BINARY word (8 lanes per ZMM)
	const __m512i absMask = _mm512_set1_epi64((long long)0x7fffffffffffffffll);
	const __m512i infBits = _mm512_set1_epi64((long long)0x7ff0000000000000ll);
	for (; i + 64 <= n; i += 64) {
		uint64_t word = 0;
		for (int chunk = 0; chunk < 8; ++chunk) {
			__m512d v = _mm512_loadu_pd(src + i + chunk * 8);
			__m512i bits = _mm512_castpd_si512(v);
			__m512i absv = _mm512_and_si512(bits, absMask);
			__mmask8 m;
			switch (cls) {
				case FloatClass::Finite:
					m = _mm512_cmplt_epu64_mask(absv, infBits);
					break;
				case FloatClass::Infinite:
					m = _mm512_cmpeq_epi64_mask(absv, infBits);
					break;
				case FloatClass::NaN:
					m = _mm512_cmpgt_epu64_mask(absv, infBits);
					break;
			}
			word |= (uint64_t)m << (chunk * 8);
		}
		outWords[i / 64] = word;
	}
#endif
	for (; i < n; ) {
		const size_t wordIndex = i / 64;
		const size_t base = wordIndex * 64;
		const size_t count = n - base < 64 ? n - base : 64;
		uint64_t word = 0;
		for (size_t j = 0; j < count; ++j) {
			uint64_t u;
			std::memcpy(&u, src + base + j, sizeof(u));
			if (f64_match(u, cls))
				word |= uint64_t(1) << j;
		}
		outWords[wordIndex] = word;
		i = base + count;
	}
}


NDArray NDArray::isFinite() const {
	const size_t n = numElements();
	NDArray out(shape, BINARY); // zero-filled
	if (type == F16 || type == BF16) {
		classifyHalfToBinary(out.uint64, u16, n, type, FloatClass::Finite);
	} else if (type == F32) {
		classifyF32ToBinary(out.uint64, float32, n, FloatClass::Finite);
	} else if (type == F64) {
		classifyF64ToBinary(out.uint64, float64, n, FloatClass::Finite);
	} else {
		// Integers / BINARY / INT3 / UINT*: always finite — fill words, not bits
		binaryFillOnes(out.uint64, n);
	}
	return out;
}

NDArray NDArray::isInfinite() const {
	const size_t n = numElements();
	NDArray out(shape, BINARY); // already all-zero from ctor
	if (type == F16 || type == BF16)
		classifyHalfToBinary(out.uint64, u16, n, type, FloatClass::Infinite);
	else if (type == F32)
		classifyF32ToBinary(out.uint64, float32, n, FloatClass::Infinite);
	else if (type == F64)
		classifyF64ToBinary(out.uint64, float64, n, FloatClass::Infinite);
	// else: never infinite — leave zeros
	return out;
}

NDArray NDArray::isNaN() const {
	const size_t n = numElements();
	NDArray out(shape, BINARY); // already all-zero from ctor
	if (type == F16 || type == BF16)
		classifyHalfToBinary(out.uint64, u16, n, type, FloatClass::NaN);
	else if (type == F32)
		classifyF32ToBinary(out.uint64, float32, n, FloatClass::NaN);
	else if (type == F64)
		classifyF64ToBinary(out.uint64, float64, n, FloatClass::NaN);
	// else: never NaN — leave zeros
	return out;
}

NDArray NDArray::asBinary() const {
	if (type == BINARY)
		return *this; // CoW share
	const size_t n = numElements();
	NDArray out(shape, BINARY);
	switch (type) {
		case F16:
		case BF16:
			for (size_t i = 0; i < n; ++i)
				if (ndarray_half::load(type, u16[i]) != 0.0f)
					binarySet(out.uint64, i, 1);
			break;
		case F32:
			nonzeroF32ToBinary(out.uint64, float32, n);
			break;
		case F64:
			nonzeroF64ToBinary(out.uint64, float64, n);
			break;
		case UINT8:
			nonzeroIntegralToBinary(out.uint64, uint8, n);
			break;
		case INT8:
			nonzeroIntegralToBinary(out.uint64, int8, n);
			break;
		case INT32:
			nonzeroIntegralToBinary(out.uint64, int32, n);
			break;
		case INT64:
			nonzeroIntegralToBinary(out.uint64, int64, n);
			break;
		case INT3:
			nonzeroInt3ToBinary(out.uint64, uint64, n);
			break;
		case UINT256:
			nonzeroU256ToBinary(out.uint64, uint256, n);
			break;
		case BINARY:
			break; // handled above
		default:
			throw std::runtime_error("NDArray::asBinary: invalid type");
	}
	return out;
}

NDArray NDArray::logicalNot() const {
	NDArray out = asBinary();
	// asBinary may share buffer when already BINARY — must detach before invert
	out.ensureWritable();
	binaryInvertInPlace(out.uint64, out.numElements());
	return out;
}

NDArray NDArray::inverted() const {
	return logicalNot();
}

NDArray NDArray::operator~() const {
	return logicalNot();
}

NDArray& NDArray::invert() {
	if (type != BINARY) {
		NDArray mask = asBinary();
		// invert a private copy then steal
		binaryInvertInPlace(mask.uint64, mask.numElements());
		stealFrom(mask);
		return *this;
	}
	ensureWritable();
	binaryInvertInPlace(uint64, numElements());
	return *this;
}


NDArray NDArray::operator&(const NDArray& other) const {
	requireSameShape(*this, other);
	const size_t n = numElements();
	if (type == BINARY && other.type == BINARY) {
		NDArray out(shape, BINARY);
		ndBinaryBitOpWords(out.uint64, uint64, other.uint64, n, NdBinAnd);
		return out;
	}
	NDArray left = asBinary();
	NDArray right = other.asBinary();
	left.ensureWritable();
	ndBinaryBitOpInPlace(left.uint64, right.uint64, n, NdBinAnd);
	return left;
}

NDArray NDArray::operator|(const NDArray& other) const {
	requireSameShape(*this, other);
	const size_t n = numElements();
	if (type == BINARY && other.type == BINARY) {
		NDArray out(shape, BINARY);
		ndBinaryBitOpWords(out.uint64, uint64, other.uint64, n, NdBinOr);
		return out;
	}
	NDArray left = asBinary();
	NDArray right = other.asBinary();
	left.ensureWritable();
	ndBinaryBitOpInPlace(left.uint64, right.uint64, n, NdBinOr);
	return left;
}

NDArray NDArray::operator^(const NDArray& other) const {
	requireSameShape(*this, other);
	const size_t n = numElements();
	if (type == BINARY && other.type == BINARY) {
		NDArray out(shape, BINARY);
		ndBinaryBitOpWords(out.uint64, uint64, other.uint64, n, NdBinXor);
		return out;
	}
	NDArray left = asBinary();
	NDArray right = other.asBinary();
	left.ensureWritable();
	ndBinaryBitOpInPlace(left.uint64, right.uint64, n, NdBinXor);
	return left;
}

NDArray& NDArray::logicalAnd(const NDArray& other) {
	requireSameShape(*this, other);
	// Always re-normalize *this by truthiness (INT3 value 2 → bit 1, not multi-bit).
	{
		NDArray self = asBinary();
		stealFrom(self);
	}
	if (other.type == BINARY)
		ndBinaryBitOpInPlace(uint64, other.uint64, numElements(), NdBinAnd);
	else {
		NDArray rhs = other.asBinary();
		ndBinaryBitOpInPlace(uint64, rhs.uint64, numElements(), NdBinAnd);
	}
	return *this;
}

NDArray& NDArray::logicalAnd(bool other) {
	if (!other) {
		// false && x → all false; allocate clean BINARY zeros
		NDArray z(shape, BINARY);
		stealFrom(z);
		return *this;
	}
	// true && x → truthiness of *this
	NDArray self = asBinary();
	stealFrom(self);
	return *this;
}

NDArray& NDArray::logicalOr(const NDArray& other) {
	requireSameShape(*this, other);
	{
		NDArray self = asBinary();
		stealFrom(self);
	}
	if (other.type == BINARY)
		ndBinaryBitOpInPlace(uint64, other.uint64, numElements(), NdBinOr);
	else {
		NDArray rhs = other.asBinary();
		ndBinaryBitOpInPlace(uint64, rhs.uint64, numElements(), NdBinOr);
	}
	return *this;
}

NDArray& NDArray::logicalOr(bool other) {
	if (other) {
		NDArray ones(shape, BINARY);
		binaryFillOnes(ones.uint64, ones.numElements());
		stealFrom(ones);
		return *this;
	}
	NDArray self = asBinary();
	stealFrom(self);
	return *this;
}

NDArray& NDArray::logicalXor(const NDArray& other) {
	requireSameShape(*this, other);
	{
		NDArray self = asBinary();
		stealFrom(self);
	}
	if (other.type == BINARY)
		ndBinaryBitOpInPlace(uint64, other.uint64, numElements(), NdBinXor);
	else {
		NDArray rhs = other.asBinary();
		ndBinaryBitOpInPlace(uint64, rhs.uint64, numElements(), NdBinXor);
	}
	return *this;
}

NDArray& NDArray::logicalXor(bool other) {
	if (other) {
		// x XOR true ≡ NOT x (after truthiness)
		NDArray self = asBinary();
		stealFrom(self);
		ensureWritable();
		binaryInvertInPlace(uint64, numElements());
		return *this;
	}
	// x XOR false ≡ x (normalized)
	NDArray self = asBinary();
	stealFrom(self);
	return *this;
}

// Out-of-place boolean &&/||: always truthiness-normalize both sides (not raw bits).
NDArray NDArray::operator&&(const NDArray& other) const {
	requireSameShape(*this, other);
	const size_t n = numElements();
	NDArray left = asBinary();
	NDArray right = other.asBinary();
	left.ensureWritable();
	ndBinaryBitOpInPlace(left.uint64, right.uint64, n, NdBinAnd);
	return left;
}

NDArray NDArray::operator||(const NDArray& other) const {
	requireSameShape(*this, other);
	const size_t n = numElements();
	NDArray left = asBinary();
	NDArray right = other.asBinary();
	left.ensureWritable();
	ndBinaryBitOpInPlace(left.uint64, right.uint64, n, NdBinOr);
	return left;
}

NDArray NDArray::operator!() const {
	return logicalNot();
}

NDArray& NDArray::operator&=(const NDArray& other) {
	requireSameShape(*this, other);
	if (type != BINARY) {
		NDArray self = asBinary();
		stealFrom(self);
	} else {
		ensureWritable();
	}
	if (other.type == BINARY)
		ndBinaryBitOpInPlace(uint64, other.uint64, numElements(), NdBinAnd);
	else {
		NDArray rhs = other.asBinary();
		ndBinaryBitOpInPlace(uint64, rhs.uint64, numElements(), NdBinAnd);
	}
	return *this;
}

NDArray& NDArray::operator|=(const NDArray& other) {
	requireSameShape(*this, other);
	if (type != BINARY) {
		NDArray self = asBinary();
		stealFrom(self);
	} else {
		ensureWritable();
	}
	if (other.type == BINARY)
		ndBinaryBitOpInPlace(uint64, other.uint64, numElements(), NdBinOr);
	else {
		NDArray rhs = other.asBinary();
		ndBinaryBitOpInPlace(uint64, rhs.uint64, numElements(), NdBinOr);
	}
	return *this;
}

NDArray& NDArray::operator^=(const NDArray& other) {
	requireSameShape(*this, other);
	if (type != BINARY) {
		NDArray self = asBinary();
		stealFrom(self);
	} else {
		ensureWritable();
	}
	if (other.type == BINARY)
		ndBinaryBitOpInPlace(uint64, other.uint64, numElements(), NdBinXor);
	else {
		NDArray rhs = other.asBinary();
		ndBinaryBitOpInPlace(uint64, rhs.uint64, numElements(), NdBinXor);
	}
	return *this;
}

bool NDArray::any() const {
	const size_t n = numElements();
	if (n == 0)
		return false;

	switch (type) {
		case BINARY: {
			const size_t full = n / 64;
			for (size_t w = 0; w < full; ++w)
				if (uint64[w] != 0)
					return true;
			const unsigned rem = (unsigned)(n % 64);
			if (rem && (uint64[full] & ((1ULL << rem) - 1ULL)) != 0)
				return true;
			return false;
		}
		case INT3: {
			// Non-zero if any value nibble (low 3 bits) is non-zero
			const size_t fullWords = n / 16;
			for (size_t w = 0; w < fullWords; ++w)
				if ((uint64[w] & int3_kValueMask) != 0)
					return true;
			for (size_t i = fullWords * 16; i < n; ++i)
				if (int3_get(uint64, i) != 0)
					return true;
			return false;
		}
		case INT8:
		case UINT8: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			for (; i + 64 <= n; i += 64) {
				__m512i v = _mm512_loadu_si512(uint8 + i);
				if (_mm512_cmpneq_epi8_mask(v, _mm512_setzero_si512()) != 0)
					return true;
			}
#endif
			// 8-byte chunks
			for (; i + 8 <= n; i += 8) {
				uint64_t chunk;
				std::memcpy(&chunk, uint8 + i, 8);
				if (chunk != 0)
					return true;
			}
			for (; i < n; ++i)
				if (uint8[i] != 0)
					return true;
			return false;
		}
		case INT32: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			for (; i + 16 <= n; i += 16) {
				__m512i v = _mm512_loadu_si512(int32 + i);
				if (_mm512_cmpneq_epi32_mask(v, _mm512_setzero_si512()) != 0)
					return true;
			}
#endif
			for (; i < n; ++i)
				if (int32[i] != 0)
					return true;
			return false;
		}
		case INT64: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			for (; i + 8 <= n; i += 8) {
				__m512i v = _mm512_loadu_si512(int64 + i);
				if (_mm512_cmpneq_epi64_mask(v, _mm512_setzero_si512()) != 0)
					return true;
			}
#endif
			for (; i < n; ++i)
				if (int64[i] != 0)
					return true;
			return false;
		}
		case F16:
		case BF16:
			for (size_t i = 0; i < n; ++i)
				if ((u16[i] & 0x7fffu) != 0)
					return true;
			return false;
		case F32: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			const __m512 z = _mm512_setzero_ps();
			for (; i + 16 <= n; i += 16) {
				__m512 v = _mm512_loadu_ps(float32 + i);
				// NEQ_UQ: NaN counts as non-zero (matches prior semantics)
				if (_mm512_cmp_ps_mask(v, z, _CMP_NEQ_UQ) != 0)
					return true;
			}
#endif
			for (; i < n; ++i)
				if (float32[i] != 0.0f)
					return true;
			return false;
		}
		case F64: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			const __m512d z = _mm512_setzero_pd();
			for (; i + 8 <= n; i += 8) {
				__m512d v = _mm512_loadu_pd(float64 + i);
				if (_mm512_cmp_pd_mask(v, z, _CMP_NEQ_UQ) != 0)
					return true;
			}
#endif
			for (; i < n; ++i)
				if (float64[i] != 0.0)
					return true;
			return false;
		}
		case UINT256: {
			const uint256_t z(0);
			for (size_t i = 0; i < n; ++i)
				if (!(uint256[i] == z))
					return true;
			return false;
		}
		default:
			throw std::runtime_error("NDArray::any: invalid type");
	}
}

bool NDArray::all() const {
	const size_t n = numElements();
	if (n == 0)
		return true;

	switch (type) {
		case BINARY: {
			const size_t full = n / 64;
			for (size_t w = 0; w < full; ++w)
				if (uint64[w] != ~uint64_t(0))
					return false;
			const unsigned rem = (unsigned)(n % 64);
			if (rem) {
				const uint64_t mask = (1ULL << rem) - 1ULL;
				if ((uint64[full] & mask) != mask)
					return false;
			}
			return true;
		}
		case INT3: {
			// Every used nibble must have non-zero value bits
			for (size_t i = 0; i < n; ++i)
				if (int3_get(uint64, i) == 0)
					return false;
			return true;
		}
		case INT8:
		case UINT8: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			for (; i + 64 <= n; i += 64) {
				__m512i v = _mm512_loadu_si512(uint8 + i);
				// any zero byte → fail
				if (_mm512_cmpeq_epi8_mask(v, _mm512_setzero_si512()) != 0)
					return false;
			}
#endif
			for (; i + 8 <= n; i += 8) {
				uint64_t chunk;
				std::memcpy(&chunk, uint8 + i, 8);
				// has zero byte: (chunk - ones) & ~chunk & 0x8080...
				const uint64_t ones = 0x0101010101010101ULL;
				const uint64_t high = 0x8080808080808080ULL;
				if (((chunk - ones) & ~chunk & high) != 0)
					return false;
			}
			for (; i < n; ++i)
				if (uint8[i] == 0)
					return false;
			return true;
		}
		case INT32: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			for (; i + 16 <= n; i += 16) {
				__m512i v = _mm512_loadu_si512(int32 + i);
				if (_mm512_cmpeq_epi32_mask(v, _mm512_setzero_si512()) != 0)
					return false;
			}
#endif
			for (; i < n; ++i)
				if (int32[i] == 0)
					return false;
			return true;
		}
		case INT64: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			for (; i + 8 <= n; i += 8) {
				__m512i v = _mm512_loadu_si512(int64 + i);
				if (_mm512_cmpeq_epi64_mask(v, _mm512_setzero_si512()) != 0)
					return false;
			}
#endif
			for (; i < n; ++i)
				if (int64[i] == 0)
					return false;
			return true;
		}
		case F16:
		case BF16:
			for (size_t i = 0; i < n; ++i)
				if ((u16[i] & 0x7fffu) == 0)
					return false;
			return true;
		case F32: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			const __m512 z = _mm512_setzero_ps();
			for (; i + 16 <= n; i += 16) {
				__m512 v = _mm512_loadu_ps(float32 + i);
				// EQ_OQ: only true zeros (NaN is not zero → counts as true)
				if (_mm512_cmp_ps_mask(v, z, _CMP_EQ_OQ) != 0)
					return false;
			}
#endif
			for (; i < n; ++i)
				if (float32[i] == 0.0f)
					return false;
			return true;
		}
		case F64: {
			size_t i = 0;
#if LIBEXCESSIVE_NDARRAY_AVX512
			const __m512d z = _mm512_setzero_pd();
			for (; i + 8 <= n; i += 8) {
				__m512d v = _mm512_loadu_pd(float64 + i);
				if (_mm512_cmp_pd_mask(v, z, _CMP_EQ_OQ) != 0)
					return false;
			}
#endif
			for (; i < n; ++i)
				if (float64[i] == 0.0)
					return false;
			return true;
		}
		case UINT256: {
			const uint256_t z(0);
			for (size_t i = 0; i < n; ++i)
				if (uint256[i] == z)
					return false;
			return true;
		}
		default:
			throw std::runtime_error("NDArray::all: invalid type");
	}
}

static int popcnt64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
	return __builtin_popcountll(x);
#else
	int n = 0;
	while (x) {
		n += (int)(x & 1u);
		x >>= 1;
	}
	return n;
#endif
}

int NDArray::countNonzero() const {
	int total = 0;

	if (type == BINARY) {
		for (size_t i = 0; i < memorySize / 8; i++)
			total += popcnt64(uint64[i]);
		return total;
	} else if (type == INT3 /* || type == INT4 || type == UINT3 || type == UINT4*/) {
		for (size_t i = 0; i < memorySize / 8; i++) {
			uint64_t tmp = uint64[i];
			tmp = (tmp & 0x3333333333333333ULL) | ((tmp >> 2) & 0x3333333333333333ULL);
			tmp = (tmp & 0x1111111111111111ULL) | ((tmp >> 1) & 0x1111111111111111ULL);
			total += popcnt64(tmp);
		}
	} else if (type == UINT8 || type == INT8)
		for (size_t i = 0; i < memorySize; i++)
			total += uint8[i] != 0 ? 1 : 0;
	else if (type == INT32)
		for (size_t i = 0; i < memorySize / 4; i++)
			total += int32[i] != 0 ? 1 : 0;
	else if (type == INT64)
		for (size_t i = 0; i < memorySize / 8; i++)
			total += int64[i] != 0 ? 1 : 0;
	else if (type == F16 || type == BF16)
		for (size_t i = 0; i < numElements(); i++)
			total += (u16[i] & 0x7fffu) != 0 ? 1 : 0;
	else if (type == F32)
		for (size_t i = 0; i < memorySize / 4; i++)
			total += float32[i] != 0 ? 1 : 0;
	else if (type == F64)
		for (size_t i = 0; i < memorySize / 8; i++)
			total += float64[i] != 0 ? 1 : 0;
	else if (type == UINT256)
		for (size_t i = 0; i < memorySize / sizeof(uint256_t); i++)
			total += uint256[i].isZero() ? 0 : 1;

	return total;
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

static int cmpViewOffsets(const NDArrayView& v, size_t ea, size_t eb) {
	const void* d = v.data();
	if (!d)
		throw std::invalid_argument("NDArray: cannot argmin/argmax an empty array");
	switch (v.getType()) {
		case F16:
		case BF16: {
			float x = ndarray_half::load(v.getType(), ((const uint16_t*)d)[ea]);
			float y = ndarray_half::load(v.getType(), ((const uint16_t*)d)[eb]);
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case F32: {
			float x = ((const float*)d)[ea], y = ((const float*)d)[eb];
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case F64: {
			double x = ((const double*)d)[ea], y = ((const double*)d)[eb];
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case UINT8: {
			uint8_t x = ((const uint8_t*)d)[ea], y = ((const uint8_t*)d)[eb];
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case INT8: {
			int8_t x = ((const int8_t*)d)[ea], y = ((const int8_t*)d)[eb];
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case INT32: {
			int32_t x = ((const int32_t*)d)[ea], y = ((const int32_t*)d)[eb];
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case INT64: {
			int64_t x = ((const int64_t*)d)[ea], y = ((const int64_t*)d)[eb];
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case INT3: {
			int x = int3_getSigned((const uint64_t*)d, ea);
			int y = int3_getSigned((const uint64_t*)d, eb);
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case BINARY: {
			uint8_t x = binaryGet((const uint64_t*)d, ea);
			uint8_t y = binaryGet((const uint64_t*)d, eb);
			return x < y ? -1 : y < x ? 1 : 0;
		}
		case UINT256: {
			const uint256_t& x = ((const uint256_t*)d)[ea];
			const uint256_t& y = ((const uint256_t*)d)[eb];
			if (x < y) return -1;
			if (y < x) return 1;
			return 0;
		}
		default:
			throw std::runtime_error("NDArray: invalid type in argmin/argmax");
	}
}

static size_t viewOffsetAlongAxis(const NDArrayView& v, int axis, size_t outerFlat, size_t along) {
	const ArrayList<int>& sh = v.getShape();
	const ArrayList<size_t>& st = v.getStrides();
	size_t o = v.getOffset();
	size_t rem = outerFlat;
	for (int d = sh.size() - 1; d >= 0; --d) {
		if (d == axis) {
			o += along * (d < st.size() ? st.get(d) : (size_t)0);
			continue;
		}
		int dim = sh.get(d);
		size_t c = dim > 0 ? rem % (size_t)dim : 0;
		if (dim > 0)
			rem /= (size_t)dim;
		o += c * (d < st.size() ? st.get(d) : (size_t)0);
	}
	return o;
}

static NDArray argExtremumAll(const NDArrayView& v, bool wantMin) {
	const size_t n = v.numElements();
	if (n == 0)
		throw std::invalid_argument("NDArray: cannot argmin/argmax an empty array");
	size_t bestI = 0;
	if (v.isContiguous() && packedArgExt(v.getType(), v.data(), v.getOffset(), n, wantMin, &bestI)) {
		NDArray out({}, INT64);
		out.set({}, (int64_t)bestI);
		return out;
	}
	size_t bestE = v.elementOffset(0);
	for (size_t i = 1; i < n; ++i) {
		size_t e = v.elementOffset(i);
		int c = cmpViewOffsets(v, e, bestE);
		if (wantMin ? c < 0 : c > 0) {
			bestE = e;
			bestI = i;
		}
	}
	NDArray out({}, INT64);
	out.set({}, (int64_t)bestI);
	return out;
}

static NDArray argExtremumAxis(const NDArrayView& v, int axis, bool wantMin) {
	const ArrayList<int>& sh = v.getShape();
	if (sh.size() == 0)
		throw std::invalid_argument("NDArray: cannot reduce a scalar along an axis");
	if (axis < 0 || axis >= sh.size())
		throw std::out_of_range("NDArray: reduction axis out of range");
	const size_t reduced = axisLength(sh, axis);
	if (reduced == 0)
		throw std::invalid_argument("NDArray: cannot reduce along a zero-length axis");
	ArrayList<int> outShape = shapeWithoutAxis(sh, axis);
	const size_t outN = resultElemsWithoutAxis(sh, axis);
	NDArray out(outShape, INT64);
	const ArrayList<size_t>& st = v.getStrides();
	const bool unitAxis = axis < st.size() && st.get(axis) == 1;
	const void* data = v.data();
	for (size_t oi = 0; oi < outN; ++oi) {
		size_t bestR = 0;
		size_t bestE = viewOffsetAlongAxis(v, axis, oi, 0);
		if (unitAxis && packedArgExt(v.getType(), data, bestE, reduced, wantMin, &bestR)) {
			out.setFlat(oi, (int64_t)bestR);
			continue;
		}
		for (size_t r = 1; r < reduced; ++r) {
			size_t e = viewOffsetAlongAxis(v, axis, oi, r);
			int c = cmpViewOffsets(v, e, bestE);
			if (wantMin ? c < 0 : c > 0) {
				bestE = e;
				bestR = r;
			}
		}
		out.setFlat(oi, (int64_t)bestR);
	}
	return out;
}

NDArray NDArrayView::argmin() const { return argExtremumAll(*this, true); }
NDArray NDArrayView::argmin(int axis) const { return argExtremumAxis(*this, axis, true); }
NDArray NDArrayView::argmax() const { return argExtremumAll(*this, false); }
NDArray NDArrayView::argmax(int axis) const { return argExtremumAxis(*this, axis, false); }

NDArray NDArray::argmin() const { return view().argmin(); }
NDArray NDArray::argmin(int axis) const { return view().argmin(axis); }
NDArray NDArray::argmax() const { return view().argmax(); }
NDArray NDArray::argmax(int axis) const { return view().argmax(axis); }


// ---- left-hand scalar operators:  scalar ⊕ NDArray -------------------------

namespace {

/** Dense array of `value` with the same shape as `ref`, starting in `seedType`. */
NDArray fullLike(const NDArray& ref, NDArrayType seedType, double value) {
	NDArray out(ref.shape, seedType);
	// zero-initialized; add scalar (may promote further via compound op)
	if (seedType == F32)
		out += (float)value;
	else if (seedType == F64)
		out += value;
	else if (seedType == INT32)
		out += (int)value;
	else if (seedType == INT64)
		out += (int64_t)value;
	else
		out += (float)value;
	return out;
}

NDArray fullLikeInt64(const NDArray& ref, int64_t value) {
	ArrayList<int64_t> data;
	data.addCopies(value, (int)ref.numElements());
	return NDArray(ref.shape, data);
}

} // namespace

NDArray operator+(float lhs, const NDArray& rhs) { return rhs + lhs; }
NDArray operator*(float lhs, const NDArray& rhs) { return rhs * lhs; }
NDArray operator-(float lhs, const NDArray& rhs) { return -rhs + lhs; }
NDArray operator/(float lhs, const NDArray& rhs) { return fullLike(rhs, F32, lhs) / rhs; }
NDArray operator%(float lhs, const NDArray& rhs) { return fullLike(rhs, F32, lhs) % rhs; }

NDArray operator+(double lhs, const NDArray& rhs) { return rhs + lhs; }
NDArray operator*(double lhs, const NDArray& rhs) { return rhs * lhs; }
NDArray operator-(double lhs, const NDArray& rhs) { return -rhs + lhs; }
NDArray operator/(double lhs, const NDArray& rhs) { return fullLike(rhs, F64, lhs) / rhs; }
NDArray operator%(double lhs, const NDArray& rhs) { return fullLike(rhs, F64, lhs) % rhs; }

NDArray operator+(int lhs, const NDArray& rhs) { return rhs + lhs; }
NDArray operator*(int lhs, const NDArray& rhs) { return rhs * lhs; }
NDArray operator-(int lhs, const NDArray& rhs) { return -rhs + lhs; }
NDArray operator/(int lhs, const NDArray& rhs) { return fullLike(rhs, INT32, lhs) / rhs; }
NDArray operator%(int lhs, const NDArray& rhs) { return fullLike(rhs, INT32, lhs) % rhs; }

NDArray operator+(int64_t lhs, const NDArray& rhs) { return rhs + lhs; }
NDArray operator*(int64_t lhs, const NDArray& rhs) { return rhs * lhs; }
NDArray operator-(int64_t lhs, const NDArray& rhs) { return -rhs + lhs; }
NDArray operator/(int64_t lhs, const NDArray& rhs) { return fullLikeInt64(rhs, lhs) / rhs; }
NDArray operator%(int64_t lhs, const NDArray& rhs) { return fullLikeInt64(rhs, lhs) % rhs; }

// Left-hand scalar comparisons:  3 > a  ≡  a < 3;  0.0 == vb  ≡  vb == 0.0
NDArray operator==(float lhs, const NDArray& rhs) { return rhs == lhs; }
NDArray operator!=(float lhs, const NDArray& rhs) { return rhs != lhs; }
NDArray operator>(float lhs, const NDArray& rhs) { return rhs < lhs; }
NDArray operator<(float lhs, const NDArray& rhs) { return rhs > lhs; }
NDArray operator>=(float lhs, const NDArray& rhs) { return rhs <= lhs; }
NDArray operator<=(float lhs, const NDArray& rhs) { return rhs >= lhs; }

NDArray operator==(double lhs, const NDArray& rhs) { return rhs == lhs; }
NDArray operator!=(double lhs, const NDArray& rhs) { return rhs != lhs; }
NDArray operator>(double lhs, const NDArray& rhs) { return rhs < lhs; }
NDArray operator<(double lhs, const NDArray& rhs) { return rhs > lhs; }
NDArray operator>=(double lhs, const NDArray& rhs) { return rhs <= lhs; }
NDArray operator<=(double lhs, const NDArray& rhs) { return rhs >= lhs; }

NDArray operator==(int lhs, const NDArray& rhs) { return rhs == lhs; }
NDArray operator!=(int lhs, const NDArray& rhs) { return rhs != lhs; }
NDArray operator>(int lhs, const NDArray& rhs) { return rhs < lhs; }
NDArray operator<(int lhs, const NDArray& rhs) { return rhs > lhs; }
NDArray operator>=(int lhs, const NDArray& rhs) { return rhs <= lhs; }
NDArray operator<=(int lhs, const NDArray& rhs) { return rhs >= lhs; }
