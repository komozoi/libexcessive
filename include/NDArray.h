
// Copyright 2021-2026 Komozoi
// Original Creation Date: 2026-7-15
//
// All rights reserved.  Do not copy, distribute, or execute, in compiled or source form,
// any portion of this software, where governed by this license.
// If you find this software, please let me know: komozoi@protonmail.com
//
//

#ifndef LIBSOLVE_NDARRAY_H
#define LIBSOLVE_NDARRAY_H

#include "bigint.h"
#include "stdint.h"
#include <type_traits>

#include "alloc/pointer.h"
#include "ds/ArrayList.h"


// Some types are commented out as not yet supported
enum NDArrayType {
	// Int types
	BINARY = 0x00,
	/** Signed 3-bit (-4..3), nibble-packed (16 values / uint64). */
	INT3 = 0x05,
	//INT4 = 0x06,
	//UINT4 = 0x07,
	//INT8 = 0x0E,
	UINT8 = 0x0F,
	//UINT16 = 0x12,
	//INT16 = 0x13,
	//UINT32 = 0x16,
	INT32 = 0x17,
	INT64 = 0x1E,
	//UINT64 = 0x1F,

	// Float types
	//F4 = 0x44,
	//F8 = 0x48,
	//F16 = 0x4A,
	F32 = 0x4C,
	F64 = 0x4E,

	//BIGINT = 0x80
	UINT256 = 0x88
};


/**
 * Shared contiguous buffer for NDArray / NDArrayView (managed via sp<> CoW).
 * Deep-copyable so COPY_ON_WRITE detach works.
 */
struct NDArrayBuffer {
	void* data = nullptr;
	size_t byteSize = 0;
	NDArrayType elementType = F32;

	NDArrayBuffer() = default;
	NDArrayBuffer(size_t bytes, NDArrayType t);
	NDArrayBuffer(const NDArrayBuffer& other);
	NDArrayBuffer(NDArrayBuffer&& other) noexcept;
	NDArrayBuffer& operator=(const NDArrayBuffer& other);
	NDArrayBuffer& operator=(NDArrayBuffer&& other) noexcept;
	~NDArrayBuffer();
};


class NDArray;
class NDArrayView;


/** Mutable chained index proxy: a[i][j] = v / T(a[i][j]). Full rank required for R/W. */
class NDArrayRef {
public:
	NDArrayRef(NDArray* parent, ArrayList<int> indices);
	NDArrayRef operator[](int i);

	template <typename T>
	operator T() const;

	template <typename T>
	NDArrayRef& operator=(const T& value);

private:
	NDArray* parent;
	ArrayList<int> indices;
};


/** Const chained index proxy. */
class NDArrayCRef {
public:
	NDArrayCRef(const NDArray* parent, ArrayList<int> indices);
	NDArrayCRef operator[](int i) const;

	template <typename T>
	operator T() const;

private:
	const NDArray* parent;
	ArrayList<int> indices;
};


/**
 * Non-owning (shared-buffer) view with independent shape/strides.
 * Survives destruction of the original NDArray via sp<> refcount.
 * Primarily for read + broadcast + materialise; in-place math materialises first.
 */
class NDArrayView {
public:
	ArrayList<int> shape;
	ArrayList<size_t> strides; // element strides; 0 = broadcast dimension
	size_t offset = 0;         // element offset into shared buffer
	NDArrayType type = F32;

	NDArrayView() = default;

	size_t numElements() const;
	bool isContiguous() const;
	bool isBroadcastableTo(const ArrayList<int>& targetShape) const;
	NDArrayView broadcastTo(const ArrayList<int>& targetShape) const;
	/** Contiguous reshape when product matches; otherwise throws. */
	NDArrayView reshape(const ArrayList<int>& newShape) const;
	NDArray copy() const;

	template <typename T>
	T get(const ArrayList<int>& indices) const;
	template <typename T>
	T getFlat(size_t flat) const;

	size_t computeOffset(const ArrayList<int>& indices) const;

	/** Shared storage (refcount); used by ndarray_detail load helpers. */
	const sp<NDArrayBuffer>& sharedBuffer() const { return buffer; }

private:
	friend class NDArray;
	sp<NDArrayBuffer> buffer;

	NDArrayView(sp<NDArrayBuffer> buf, ArrayList<int> shape, ArrayList<size_t> strides,
	            size_t offset, NDArrayType type);
};


class NDArray {
public:
	NDArray(ArrayList<int> shape, NDArrayType type);
	NDArray(ArrayList<float> vector);
	NDArray(ArrayList<double> vector);
	NDArray(ArrayList<uint8_t> vector);
	NDArray(ArrayList<int32_t> vector);
	NDArray(ArrayList<int64_t> vector);
	NDArray(const ArrayList<int>& shape, ArrayList<float> vector);
	NDArray(const ArrayList<int>& shape, ArrayList<double> vector);
	NDArray(const ArrayList<int>& shape, ArrayList<uint8_t> vector);
	NDArray(const ArrayList<int>& shape, ArrayList<int32_t> vector);
	NDArray(const ArrayList<int>& shape, ArrayList<int64_t> vector);

	// Convenience constructors for scalars
	template <typename T, typename = std::enable_if_t<std::is_same_v<T, NDArrayType>>>
	NDArray(T type, float value) : shape({}), type(static_cast<NDArrayType>(type)) {
		initialize();
		set({}, value);
	}

	template <typename T, typename = std::enable_if_t<std::is_same_v<T, NDArrayType>>>
	NDArray(T type, double value) : shape({}), type(static_cast<NDArrayType>(type)) {
		initialize();
		set({}, value);
	}

	template <typename T, typename = std::enable_if_t<std::is_same_v<T, NDArrayType>>>
	NDArray(T type, int value) : shape({}), type(static_cast<NDArrayType>(type)) {
		initialize();
		set({}, value);
	}

	NDArray(const NDArray& other);
	NDArray(NDArray&& other) noexcept;
	~NDArray();

	// CoW: share buffer on copy/assign; deep-copy on write via buffer.mut().
	NDArray& operator=(const NDArray& other);
	NDArray& operator=(NDArray&& other) noexcept;

	bool operator==(const NDArray& nds) const;

	/**
	 * Explicitly convert this array to a new element type.
	 *
	 * This is the only API that is allowed to drop precision / range: the
	 * programmer is asking for the conversion. Element values are cast
	 * according to the source and destination types.
	 *
	 * @param newType destination element type
	 * @return a new NDArray with the same shape and converted contents
	 */
	NDArray convert(NDArrayType newType) const;

	/** Number of logical elements (product of shape; 1 for a scalar). */
	size_t numElements() const;

	/** Row-major element strides for this dense array. */
	ArrayList<size_t> strides() const;
	bool isContiguous() const { return true; } // owners are dense
	bool isBroadcastableTo(const ArrayList<int>& targetShape) const;
	NDArrayView view() const;
	NDArrayView broadcastTo(const ArrayList<int>& targetShape) const;
	NDArrayView reshapeView(const ArrayList<int>& newShape) const;

	NDArrayRef operator[](int i);
	NDArrayCRef operator[](int i) const;

	/** Runtime multi-index (rank = shape.size()). */
	template <typename T>
	T get(const ArrayList<int>& indices) const {
		if (indices.size() != shape.size())
			throw std::out_of_range("NDArray::get - rank mismatch");
		return getAtOffset<T>(computeOffset(indices));
	}

	template <typename T>
	void set(const ArrayList<int>& indices, const T& value) {
		if (indices.size() != shape.size())
			throw std::out_of_range("NDArray::set - rank mismatch");
		setAtOffset(computeOffset(indices), value);
	}

	template <typename T>
	T get(std::initializer_list<int> indices) const {
		return get<T>(ArrayList<int>(indices));
	}

	template <typename T>
	void set(std::initializer_list<int> indices, const T& value) {
		set(ArrayList<int>(indices), value);
	}

	/** Flat element index in dense row-major order. */
	template <typename T>
	T getFlat(size_t flat) const {
		if (flat >= numElements())
			throw std::out_of_range("NDArray::getFlat - index out of range");
		return getAtOffset<T>(flat);
	}

	template <typename T>
	void setFlat(size_t flat, const T& value) {
		if (flat >= numElements())
			throw std::out_of_range("NDArray::setFlat - index out of range");
		setAtOffset(flat, value);
	}

	/*
	** Mathematical operations (operator overloads also present, these are the basic forms)
	*/

	/**
	 * Adds two NDArrays of the same shape element-wise.
	 *
	 * Updates `*this` in place.  If the result type must be wider than
	 * `this->type` to avoid losing precision, `*this` is promoted (contents
	 * converted) first.  Does not mutate `other`.
	 *
	 * @param other array to add
	 * @return *this
	 */
	NDArray& add(const NDArray& other);

	/**
	 * Subtracts `other` from `*this` element-wise (same shape).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& sub(const NDArray& other);

	/**
	 * Multiplies two NDArrays of the same shape element-wise.
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& mul(const NDArray& other);

	/**
	 * Divides `*this` by `other` element-wise (same shape).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& div(const NDArray& other);

	/**
	 * Adds `other` into `*this` with broadcasting.
	 *
	 * `other.shape` must be a prefix of `this->shape`.  Each element of
	 * `other` is applied to a contiguous block of `*this` corresponding to
	 * the remaining (trailing) dimensions.
	 *
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 *
	 * @param other array to add (broadcast along outer axes)
	 * @return *this
	 */
	NDArray& broadcastAdd(const NDArray& other);

	/**
	 * Subtracts a broadcasted `other` from `*this` (prefix-shape rule).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& broadcastSub(const NDArray& other);

	/**
	 * Multiplies `*this` by a broadcasted `other` (prefix-shape rule).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& broadcastMul(const NDArray& other);

	/**
	 * Divides `*this` by a broadcasted `other` (prefix-shape rule).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& broadcastDiv(const NDArray& other);

	/*
	** Factories (return new arrays)
	*/
	static NDArray zeros(const ArrayList<int>& shape, NDArrayType type);
	static NDArray ones(const ArrayList<int>& shape, NDArrayType type);
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, float value);
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, double value);
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, int value);
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, int64_t value);
	static NDArray zerosLike(const NDArray& ref);
	static NDArray onesLike(const NDArray& ref);
	static NDArray fullLike(const NDArray& ref, float value);
	static NDArray fullLike(const NDArray& ref, double value);
	static NDArray fullLike(const NDArray& ref, int value);
	static NDArray fullLike(const NDArray& ref, int64_t value);

	/*
	** Element-wise unary
	**
	** Mutators (return *this, change storage in place).  Example:
	**   b.square().square();  // b becomes b^4
	**
	** Out-of-place helpers copy first (safe in expressions).  Binary
	** operators always return new arrays.
	*/
	NDArray& neg();
	NDArray& abs();
	NDArray& sign();
	NDArray& square();
	NDArray& sqrt();
	NDArray& cbrt();
	NDArray& exp();
	NDArray& expm1();
	NDArray& log();
	NDArray& log2();
	NDArray& log10();
	NDArray& log1p();
	// Trigonometric (radians), inverse, hyperbolic — in place; promote to F32/F64 as needed
	NDArray& sin();
	NDArray& cos();
	NDArray& tan();
	NDArray& asin();
	NDArray& acos();
	NDArray& atan();
	NDArray& sinh();
	NDArray& cosh();
	NDArray& tanh();
	NDArray& asinh();
	NDArray& acosh();
	NDArray& atanh();
	/** Degrees ↔ radians (in place). */
	NDArray& deg2rad();
	NDArray& rad2deg();

	NDArray& floor();
	NDArray& ceil();
	NDArray& round();

	/** Out-of-place: copy then square / neg / abs. */
	NDArray squared() const;
	NDArray negated() const;
	NDArray absolute() const;

	/**
	 * Two-argument arctangent: atan2(*this, x) element-wise (returns new array).
	 * Same convention as std::atan2(y, x) with *this as y.
	 */
	NDArray atan2(const NDArray& x) const;
	NDArray atan2(double x) const;
	NDArray atan2(float x) const;
	NDArray atan2(int x) const;

	/** Hypotenuse: hypot(*this, x) = sqrt(this² + x²), returns new array. */
	NDArray hypot(const NDArray& x) const;
	NDArray hypot(double x) const;
	NDArray hypot(float x) const;
	NDArray hypot(int x) const;

	/** Unary minus — returns a new array (operators copy). */
	NDArray operator-() const;

	/*
	** Element-wise binary (return a new array; same promote-on-op rules as + / *)
	*/
	NDArray minimum(const NDArray& other) const;
	NDArray maximum(const NDArray& other) const;
	NDArray pow(const NDArray& other) const;
	NDArray mod(const NDArray& other) const;
	/** Clamp each element into [lo, hi] (promotes as needed). */
	NDArray clip(const NDArray& lo, const NDArray& hi) const;
	NDArray clip(float lo, float hi) const;
	NDArray clip(double lo, double hi) const;

	NDArray minimum(float other) const;
	NDArray maximum(float other) const;
	NDArray pow(float other) const;
	NDArray mod(float other) const;

	NDArray minimum(double other) const;
	NDArray maximum(double other) const;
	NDArray pow(double other) const;
	NDArray mod(double other) const;

	NDArray minimum(int other) const;
	NDArray maximum(int other) const;
	NDArray pow(int other) const;
	NDArray mod(int other) const;

	NDArray minimum(int64_t other) const;
	NDArray maximum(int64_t other) const;
	NDArray pow(int64_t other) const;
	NDArray mod(int64_t other) const;

	/*
	** Element-wise comparisons → BINARY mask (same shape).
	**
	** Whole-array equality: operator==(const NDArray&) → bool.
	** Element-wise vs scalar: operator==(float|double|int) → BINARY, e.g. vb == 0.0
	** Element-wise relational: >, <, >=, <=, != → BINARY
	**
	** Ternary-style selection (C++ cannot overload ?:):
	**   select(a > b, vb, va);
	**   (a > b).choose(vb, va);
	*/
	/** Element-wise array comparison (BINARY). Prefer operators for scalars. */
	NDArray equal(const NDArray& other) const;
	NDArray notEqual(const NDArray& other) const;
	NDArray less(const NDArray& other) const;
	NDArray lessEqual(const NDArray& other) const;
	NDArray greater(const NDArray& other) const;
	NDArray greaterEqual(const NDArray& other) const;

	/** Element-wise relational operators → BINARY (not whole-array). */
	NDArray operator>(const NDArray& other) const;
	NDArray operator<(const NDArray& other) const;
	NDArray operator>=(const NDArray& other) const;
	NDArray operator<=(const NDArray& other) const;
	NDArray operator!=(const NDArray& other) const;

	/** Element-wise equality / inequality vs scalar → BINARY. */
	NDArray operator==(float other) const;
	NDArray operator!=(float other) const;
	NDArray operator>(float other) const;
	NDArray operator<(float other) const;
	NDArray operator>=(float other) const;
	NDArray operator<=(float other) const;

	NDArray operator==(double other) const;
	NDArray operator!=(double other) const;
	NDArray operator>(double other) const;
	NDArray operator<(double other) const;
	NDArray operator>=(double other) const;
	NDArray operator<=(double other) const;

	NDArray operator==(int other) const;
	NDArray operator!=(int other) const;
	NDArray operator>(int other) const;
	NDArray operator<(int other) const;
	NDArray operator>=(int other) const;
	NDArray operator<=(int other) const;

	/**
	 * NDArray or scalar (int / float / double / int64_t) for value slots in
	 * select, choose, safeDiv, piecewise, etc. Scalars are written into the
	 * result only when chosen — no full temporary array is allocated for them.
	 */
	class ArrayOrScalar {
	public:
		ArrayOrScalar(const NDArray& a);
		ArrayOrScalar(double v);
		ArrayOrScalar(float v);
		ArrayOrScalar(int v);
		ArrayOrScalar(int64_t v);

	private:
		friend class NDArray;
		friend struct Impl;
		enum Kind { Array, ScalarDouble, ScalarInt };
		Kind kind;
		const NDArray* arr;
		double d;
		int64_t i;
	};

	/**
	 * Element-wise select: result[i] = condition[i] ? x[i] : y[i].
	 *
	 * x and y may be NDArray or scalars (ArrayOrScalar). Prefer safeDiv when a
	 * branch would divide by zero. C++ cannot overload `?:`; use:
	 *   select(a > b, vb, va)
	 *   (a > b).choose(vb, va)
	 */
	static NDArray where(const NDArray& condition, ArrayOrScalar x, ArrayOrScalar y);
	/** Alias of where. */
	static NDArray select(const NDArray& condition, ArrayOrScalar x, ArrayOrScalar y);

	/**
	 * Instance form of select: *this is the condition mask.
	 *   (a > b).choose(vb, va)  ≡  select(a > b, vb, va)
	 */
	NDArray choose(ArrayOrScalar ifTrue, ArrayOrScalar ifFalse) const;

	/**
	 * Element-wise *this / den, never dividing by zero:
	 *   out[i] = (den[i] == 0) ? whenZero : (*this)[i] / den[i]
	 * whenZero may be an array (same shape) or a scalar.
	 */
	NDArray safeDiv(const NDArray& den, ArrayOrScalar whenZero) const;

	/**
	 * Ordered multi-branch select (first true mask wins; last arg is default).
	 * Value arms are ArrayOrScalar (NDArray or int/float/double).
	 *
	 * Example:
	 *   piecewise(vb == 0.0, when_vb0, b, main, -1.0)
	 */
	static NDArray piecewise(const NDArray& m0, ArrayOrScalar v0, ArrayOrScalar otherwise);
	static NDArray piecewise(const NDArray& m0, ArrayOrScalar v0,
	                         const NDArray& m1, ArrayOrScalar v1, ArrayOrScalar otherwise);
	static NDArray piecewise(const NDArray& m0, ArrayOrScalar v0,
	                         const NDArray& m1, ArrayOrScalar v1,
	                         const NDArray& m2, ArrayOrScalar v2, ArrayOrScalar otherwise);

	/** True if any element is non-zero / true. */
	bool any() const;
	/** True if every element is non-zero / true. */
	bool all() const;

	/*
	** Reductions
	**
	** No-axis forms reduce to a scalar (empty shape).  Axis forms remove that
	** axis from the shape.  Unsigned integer sum/prod accumulate in UINT256;
	** INT32 sum/prod use INT64; floats keep their float type.  mean uses F64
	** except pure F32 inputs (stay F32).
	*/
	NDArray sum() const;
	NDArray sum(int axis) const;
	NDArray mean() const;
	NDArray mean(int axis) const;
	NDArray min() const;
	NDArray min(int axis) const;
	NDArray max() const;
	NDArray max(int axis) const;
	NDArray prod() const;
	NDArray prod(int axis) const;

	/*
	**    OPERATORS  (return new arrays — they copy)
	*/

	NDArray operator+(float other) const;
	NDArray operator-(float other) const;
	NDArray operator*(float other) const;
	NDArray operator/(float other) const;
	NDArray operator%(float other) const;

	NDArray operator+(int other) const;
	NDArray operator-(int other) const;
	NDArray operator*(int other) const;
	NDArray operator/(int other) const;
	NDArray operator%(int other) const;

	NDArray operator+(int64_t other) const;
	NDArray operator-(int64_t other) const;
	NDArray operator*(int64_t other) const;
	NDArray operator/(int64_t other) const;
	NDArray operator%(int64_t other) const;

	NDArray operator+(double other) const;
	NDArray operator-(double other) const;
	NDArray operator*(double other) const;
	NDArray operator/(double other) const;
	NDArray operator%(double other) const;

	NDArray operator+(const NDArray& other) const;
	NDArray operator-(const NDArray& other) const;
	NDArray operator*(const NDArray& other) const;
	NDArray operator/(const NDArray& other) const;
	/** Element-wise modulo; same as mod(). */
	NDArray operator%(const NDArray& other) const;

	NDArray& operator+=(float other);
	NDArray& operator-=(float other);
	NDArray& operator*=(float other);
	NDArray& operator/=(float other);
	NDArray& operator%=(float other);

	NDArray& operator+=(int other);
	NDArray& operator-=(int other);
	NDArray& operator*=(int other);
	NDArray& operator/=(int other);
	NDArray& operator%=(int other);

	NDArray& operator+=(int64_t other);
	NDArray& operator-=(int64_t other);
	NDArray& operator*=(int64_t other);
	NDArray& operator/=(int64_t other);
	NDArray& operator%=(int64_t other);

	NDArray& operator+=(double other);
	NDArray& operator-=(double other);
	NDArray& operator*=(double other);
	NDArray& operator/=(double other);
	NDArray& operator%=(double other);

	NDArray& operator+=(const NDArray& other);
	NDArray& operator-=(const NDArray& other);
	NDArray& operator*=(const NDArray& other);
	NDArray& operator/=(const NDArray& other);
	NDArray& operator%=(const NDArray& other);

	ArrayList<int> shape;
	// Mutable so in-place arithmetic can promote; kept consistent with buffer dtype.
	NDArrayType type;

private:
	enum class ArithOp { Add, Sub, Mul, Div };

	/** .cpp-only helpers that need access to storage pointers. */
	struct Impl;
	friend struct Impl;
	friend class NDArrayRef;
	friend class NDArrayCRef;
	friend class NDArrayView;

	size_t initialize();
	size_t computeOffset(const ArrayList<int>& indices) const;
	void rebindPointers();
	void ensureWritable();
	static ArrayList<size_t> rowMajorStrides(const ArrayList<int>& shape);
	static size_t bufferBytesFor(NDArrayType t, size_t numElems);

	template <typename T>
	T getAtOffset(size_t offset) const {
		switch (type) {
			case BINARY:
				return static_cast<T>((uint64[offset >> 6] >> (offset & 63)) & 1);
			case INT3: {
				// Signed decode -4..3 (nibble-packed)
				const size_t w = offset / 16;
				const unsigned sh = (unsigned)((offset % 16) * 4);
				uint8_t p = (uint8_t)((uint64[w] >> sh) & 7u);
				int v = (p >= 4) ? (int)p - 8 : (int)p;
				return static_cast<T>(v);
			}
			case UINT8:
				return static_cast<T>(uint8[offset]);
			case INT32:
				if constexpr (std::is_same_v<T, uint256_t>)
					return uint256_t((int)int32[offset]);
				else
					return static_cast<T>(int32[offset]);
			case INT64:
				if constexpr (std::is_same_v<T, uint256_t>)
					return int64[offset] < 0
						? uint256_t((int)int64[offset])
						: uint256_t((uint64_t)int64[offset]);
				else
					return static_cast<T>(int64[offset]);
			case F32:
				return static_cast<T>(float32[offset]);
			case F64:
				if constexpr (std::is_same_v<T, uint256_t>)
					return uint256_t(float64[offset]);
				else
					return static_cast<T>(float64[offset]);
			case UINT256:
				return convert_from_uint256<T>(uint256[offset]);
			default:
				throw std::runtime_error("NDArray::getAtOffset - invalid type");
		}
	}

	template <typename T>
	void setAtOffset(size_t offset, const T& value) {
		ensureWritable();
		if constexpr (std::is_same_v<T, uint256_t>) {
			switch (type) {
				case BINARY:
					if ((uint64_t)value != 0)
						uint64[offset >> 6] |= 1ULL << (offset & 63);
					else
						uint64[offset >> 6] &= ~(1ULL << (offset & 63));
					break;
				case INT3: {
					uint8_t p = (uint8_t)((uint64_t)value & 7u);
					const size_t w = offset / 16;
					const unsigned sh = (unsigned)((offset % 16) * 4);
					uint64[w] = (uint64[w] & ~(0xFULL << sh)) | ((uint64_t)p << sh);
					break;
				}
				case UINT8:
					uint8[offset] = (uint8_t)(uint64_t)value;
					break;
				case INT32:
					int32[offset] = (int32_t)(uint64_t)value;
					break;
				case INT64:
					int64[offset] = (int64_t)(uint64_t)value;
					break;
				case F32:
					float32[offset] = (float)value.toDouble();
					break;
				case F64:
					float64[offset] = value.toDouble();
					break;
				case UINT256:
					uint256[offset] = value;
					break;
			}
		} else {
			switch (type) {
				case BINARY:
					if (value > 0)
						uint64[offset >> 6] |= 1ULL << (offset & 63);
					else
						uint64[offset >> 6] &= ~(1ULL << (offset & 63));
					break;
				case INT3: {
					int iv;
					if constexpr (std::is_floating_point_v<T>)
						iv = (int)value;
					else
						iv = (int)value;
					uint8_t p = (uint8_t)(iv & 7);
					const size_t w = offset / 16;
					const unsigned sh = (unsigned)((offset % 16) * 4);
					uint64[w] = (uint64[w] & ~(0xFULL << sh)) | ((uint64_t)p << sh);
					break;
				}
				case UINT8:
					uint8[offset] = (uint8_t)value;
					break;
				case INT32:
					int32[offset] = (int32_t)value;
					break;
				case INT64:
					int64[offset] = (int64_t)value;
					break;
				case F32:
					float32[offset] = (float)value;
					break;
				case F64:
					float64[offset] = (double)value;
					break;
				case UINT256:
					if constexpr (std::is_integral_v<T>)
						uint256[offset] = value < 0
							? uint256_t((int)value)
							: uint256_t((uint64_t)value);
					else if constexpr (std::is_floating_point_v<T>)
						uint256[offset] = uint256_t((double)value);
					else
						uint256[offset] = uint256_t((uint64_t)0);
					break;
			}
		}
	}

	float loadAsFloat(size_t i) const;
	double loadAsDouble(size_t i) const;
	int64_t loadAsI64(size_t i) const;
	uint256_t loadAsU256(size_t i) const;
	void storeFromFloat(size_t i, float v);
	void storeFromDouble(size_t i, double v);
	void storeFromI64(size_t i, int64_t v);
	void storeFromU256(size_t i, const uint256_t& v);

	/** Convert storage in place to `newType` (no-op if already that type). */
	void promoteInPlace(NDArrayType newType);

	void applyBinaryInPlace(const NDArray& src, ArithOp op);
	void applyFloatScalarInPlace(float scalar, ArithOp op);
	void applyDoubleScalarInPlace(double scalar, ArithOp op);
	void applyIntScalarInPlace(int scalar, ArithOp op);

	/** Promote *this to a common type with `other`, then apply op. */
	NDArray& binaryOpInPlace(const NDArray& other, ArithOp op);
	/** Promote *this for a float scalar, then apply op. */
	NDArray& scalarFloatOpInPlace(float other, ArithOp op);
	/** Promote *this for a double scalar, then apply op. */
	NDArray& scalarDoubleOpInPlace(double other, ArithOp op);
	/** Promote *this for an int scalar, then apply op. */
	NDArray& scalarIntOpInPlace(int other, ArithOp op);
	/** Promote *this for an int64 scalar, then apply op. */
	NDArray& scalarInt64OpInPlace(int64_t other, ArithOp op);
	void applyInt64ScalarInPlace(int64_t scalar, ArithOp op);
	/**
	 * Broadcast `other` (shape prefix of *this) and apply op in place.
	 * Promotes *this to a common type with `other` first when needed.
	 */
	NDArray& broadcastOpInPlace(const NDArray& other, ArithOp op);
	/** Apply broadcast op; requires same type and prefix shape. */
	void applyBroadcastInPlace(const NDArray& src, ArithOp op);

	NDArray binaryOp(const NDArray& other, ArithOp op) const;
	NDArray scalarFloatOp(float other, ArithOp op) const;
	NDArray scalarDoubleOp(double other, ArithOp op) const;
	NDArray scalarIntOp(int other, ArithOp op) const;
	NDArray scalarInt64Op(int64_t other, ArithOp op) const;

	/** Promote in place to a real type and apply unary kernel (F32 or F64). */
	NDArray& mapRealUnary(double (*fn)(double));

	/** Replace storage with that of `result` (moves buffer; used by %= etc.). */
	void stealFrom(NDArray& result);

	/** Fill every element with value (cast into current type). */
	void fillFromDouble(double value);
	void fillFromI64(int64_t value);

	template <typename T>
	static T convert_from_uint256(const uint256_t& v) {
		// Exact match
		if constexpr (std::is_same_v<T, uint256_t>) {
			return v;
		}
		// LongKey<N> family – uint256_t already knows how to convert
		else if constexpr (std::is_convertible_v<uint256_t, T> &&
						   !std::is_arithmetic_v<T>) {
			return static_cast<T>(v);
						   }
		// Floating-point types
		else if constexpr (std::is_floating_point_v<T>) {
			return static_cast<T>(v.toDouble());
		}
		// Unsigned integers up to 64-bit: go through uint64_t first
		else if constexpr (std::is_unsigned_v<T> && sizeof(T) <= 8) {
			return static_cast<T>(static_cast<uint64_t>(v));
		}
		// Signed integers up to 64-bit: also go through uint64_t (then cast)
		else if constexpr (std::is_signed_v<T> && sizeof(T) <= 8) {
			return static_cast<T>(static_cast<uint64_t>(v));
		}
		else {
			static_assert(sizeof(T) == 0,
				"convert_from_uint256: unsupported target type");
		}
	}

	sp<NDArrayBuffer> buffer; // COPY_ON_WRITE shared storage
	size_t memorySize = 0;
	union {
		void* memory;

		uint8_t* uint8;
		int32_t* int32;
		int64_t* int64;

		// Used for uint64 *and* binary encoding
		uint64_t* uint64;

		float* float32;
		double* float64;

		uint256_t* uint256;
	};
};


// ---- proxy template methods (need NDArray complete) -------------------------

template <typename T>
NDArrayRef::operator T() const {
	return parent->get<T>(indices);
}

template <typename T>
NDArrayRef& NDArrayRef::operator=(const T& value) {
	parent->set(indices, value);
	return *this;
}

template <typename T>
NDArrayCRef::operator T() const {
	return parent->get<T>(indices);
}

// View element access: defined in NDArray.cpp (uses shared buffer load helpers)
// Explicit instantiations provided for common T; generic path uses convert via double/i64.
namespace ndarray_detail {
	double viewLoadDouble(const NDArrayView& v, size_t elementOffset);
	int64_t viewLoadI64(const NDArrayView& v, size_t elementOffset);
	uint256_t viewLoadU256(const NDArrayView& v, size_t elementOffset);
}

template <typename T>
T NDArrayView::getFlat(size_t flat) const {
	if (flat >= numElements())
		throw std::out_of_range("NDArrayView::getFlat - index out of range");
	// Map dense flat index of *view shape* to buffer element offset via multi-index
	ArrayList<int> idx;
	size_t rem = flat;
	// build multi-index row-major
	size_t prod = 1;
	for (int d = shape.size() - 1; d >= 0; --d)
		prod *= (size_t)(shape.get(d) > 0 ? shape.get(d) : 1);
	// simpler: iterative divide
	ArrayList<int> coords;
	for (int i = 0; i < shape.size(); ++i)
		coords.add(0);
	size_t r = flat;
	for (int d = shape.size() - 1; d >= 0; --d) {
		int dim = shape.get(d);
		int c = dim > 0 ? (int)(r % (size_t)dim) : 0;
		coords.set(d, c);
		if (dim > 0)
			r /= (size_t)dim;
	}
	size_t elemOff = computeOffset(coords);
	if constexpr (std::is_same_v<T, uint256_t>)
		return ndarray_detail::viewLoadU256(*this, elemOff);
	else if constexpr (std::is_floating_point_v<T>)
		return static_cast<T>(ndarray_detail::viewLoadDouble(*this, elemOff));
	else
		return static_cast<T>(ndarray_detail::viewLoadI64(*this, elemOff));
}

template <typename T>
T NDArrayView::get(const ArrayList<int>& indices) const {
	if (indices.size() != shape.size())
		throw std::out_of_range("NDArrayView::get - rank mismatch");
	size_t elemOff = computeOffset(indices);
	if constexpr (std::is_same_v<T, uint256_t>)
		return ndarray_detail::viewLoadU256(*this, elemOff);
	else if constexpr (std::is_floating_point_v<T>)
		return static_cast<T>(ndarray_detail::viewLoadDouble(*this, elemOff));
	else
		return static_cast<T>(ndarray_detail::viewLoadI64(*this, elemOff));
}


/*
** Left-hand scalar operators:  scalar ⊕ NDArray
** (Member operators already cover NDArray ⊕ scalar.)
*/
NDArray operator+(float lhs, const NDArray& rhs);
NDArray operator-(float lhs, const NDArray& rhs);
NDArray operator*(float lhs, const NDArray& rhs);
NDArray operator/(float lhs, const NDArray& rhs);
NDArray operator%(float lhs, const NDArray& rhs);

NDArray operator+(double lhs, const NDArray& rhs);
NDArray operator-(double lhs, const NDArray& rhs);
NDArray operator*(double lhs, const NDArray& rhs);
NDArray operator/(double lhs, const NDArray& rhs);
NDArray operator%(double lhs, const NDArray& rhs);

NDArray operator+(int lhs, const NDArray& rhs);
NDArray operator-(int lhs, const NDArray& rhs);
NDArray operator*(int lhs, const NDArray& rhs);
NDArray operator/(int lhs, const NDArray& rhs);
NDArray operator%(int lhs, const NDArray& rhs);

NDArray operator+(int64_t lhs, const NDArray& rhs);
NDArray operator-(int64_t lhs, const NDArray& rhs);
NDArray operator*(int64_t lhs, const NDArray& rhs);
NDArray operator/(int64_t lhs, const NDArray& rhs);
NDArray operator%(int64_t lhs, const NDArray& rhs);

/** Left-hand scalar comparisons → BINARY mask (element-wise). */
NDArray operator==(float lhs, const NDArray& rhs);
NDArray operator!=(float lhs, const NDArray& rhs);
NDArray operator>(float lhs, const NDArray& rhs);
NDArray operator<(float lhs, const NDArray& rhs);
NDArray operator>=(float lhs, const NDArray& rhs);
NDArray operator<=(float lhs, const NDArray& rhs);

NDArray operator==(double lhs, const NDArray& rhs);
NDArray operator!=(double lhs, const NDArray& rhs);
NDArray operator>(double lhs, const NDArray& rhs);
NDArray operator<(double lhs, const NDArray& rhs);
NDArray operator>=(double lhs, const NDArray& rhs);
NDArray operator<=(double lhs, const NDArray& rhs);

NDArray operator==(int lhs, const NDArray& rhs);
NDArray operator!=(int lhs, const NDArray& rhs);
NDArray operator>(int lhs, const NDArray& rhs);
NDArray operator<(int lhs, const NDArray& rhs);
NDArray operator>=(int lhs, const NDArray& rhs);
NDArray operator<=(int lhs, const NDArray& rhs);


#endif //AGENT_CLUSTER_NDARRAY_H
