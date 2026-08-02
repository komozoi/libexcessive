
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

#include "ds/ArrayList.h"


// Some types are commented out as not yet supported
enum NDArrayType {
	// Int types
	BINARY = 0x00,
	//INT4 = 0x06,
	//UINT4 = 0x07,
	//INT8 = 0x0E,
	UINT8 = 0x0F,
	/*UINT16 = 0x12,
	INT16 = 0x13,
	UINT32 = 0x16,
	INT32 = 0x17,
	INT64 = 0x1E,
	UINT64 = 0x1F,*/

	// Float types
	//F4 = 0x44,
	//F8 = 0x48,
	//F16 = 0x4A,
	F32 = 0x4C,
	//F64 = 0x4E

	//BIGINT = 0x80
	UINT256 = 0x88
};


class NDArray {
public:
	NDArray(ArrayList<int> shape, NDArrayType type);
	NDArray(ArrayList<float> vector);
	NDArray(ArrayList<uint8_t> vector);
	NDArray(const ArrayList<int>& shape, ArrayList<float> vector);
	NDArray(const ArrayList<int>& shape, ArrayList<uint8_t> vector);

	// Convenience constructors for scalars
	template <typename T, typename = std::enable_if_t<std::is_same_v<T, NDArrayType>>>
	NDArray(T type, float value) : shape({}), type(static_cast<NDArrayType>(type)) {
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

	// const shape: assignment would be ill-formed; copies use the copy ctor.
	// (type is mutable so in-place arithmetic can promote.)
	NDArray& operator=(const NDArray&) = delete;
	NDArray& operator=(NDArray&&) = delete;

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

	template <typename T>
	T get(std::initializer_list<int> indices) const {
		if ((int)indices.size() != shape.size())
			throw std::out_of_range("NDArray::get - number of indices did not match number of axes");

		size_t offset = computeOffset(indices);

		switch (type) {
			case BINARY:
				return static_cast<T>((uint64[offset >> 6] >> (offset & 63)) & 1);

			case UINT8:
				return static_cast<T>(uint8[offset]);

			case F32:
				return static_cast<T>(float32[offset]);

			case UINT256:
				return convert_from_uint256<T>(uint256[offset]);

			default:
				throw std::runtime_error("NDArray::get - invalid NDArray type");
		}
	}

	template <typename T>
	void set(std::initializer_list<int> indices, const T& value) {
		if ((int)indices.size() != shape.size())
			throw std::out_of_range("NDArray::set - number of indices did not match number of axes");
		size_t offset = computeOffset(indices);
		switch (type) {
			case BINARY:
				if (value > 0)
					uint64[offset >> 6] |= 1ULL << (offset & 63);
				else
					uint64[offset >> 6] &= ~(1ULL << (offset & 63));
				break;
			case UINT8:
				uint8[offset] = (uint8_t)value;
				break;
			case F32:
				float32[offset] = (float)value;
				break;
			case UINT256:
				uint256[offset] = (uint256_t)value;
				break;
		}
	}

	/*
	** Mathematical operations (operator overloads also present, these are the basic forms)
	*/

	/**
	 * Adds two NDArrays of the same shape element-wise.
	 *
	 * Updates `*this` in place.  If the result type must be wider than
	 * `this->type` to avoid losing precision, `*this` is promoted (contents
	 * converted) first.
	 *
	 * @param other array to add
	 * @return *this
	 */
	NDArray& add(NDArray& other);

	/**
	 * Subtracts `other` from `*this` element-wise (same shape).
	 * Promotes `*this` when needed so precision is not lost.
	 */
	NDArray& sub(NDArray& other);

	/**
	 * Multiplies two NDArrays of the same shape element-wise.
	 * Promotes `*this` when needed so precision is not lost.
	 */
	NDArray& mul(NDArray& other);

	/**
	 * Divides `*this` by `other` element-wise (same shape).
	 * Promotes `*this` when needed so precision is not lost.
	 */
	NDArray& div(NDArray& other);

	/**
	 * Adds `other` into `*this` with broadcasting.
	 *
	 * `other.shape` must be a prefix of `this->shape`.  Each element of
	 * `other` is applied to a contiguous block of `*this` corresponding to
	 * the remaining (trailing) dimensions.
	 *
	 * Promotes `*this` when needed so precision is not lost.
	 *
	 * @param other array to add (broadcast along outer axes)
	 * @return *this
	 */
	NDArray& broadcastAdd(NDArray& other);

	/**
	 * Subtracts a broadcasted `other` from `*this` (prefix-shape rule).
	 * Promotes `*this` when needed so precision is not lost.
	 */
	NDArray& broadcastSub(NDArray& other);

	/**
	 * Multiplies `*this` by a broadcasted `other` (prefix-shape rule).
	 * Promotes `*this` when needed so precision is not lost.
	 */
	NDArray& broadcastMul(NDArray& other);

	/**
	 * Divides `*this` by a broadcasted `other` (prefix-shape rule).
	 * Promotes `*this` when needed so precision is not lost.
	 */
	NDArray& broadcastDiv(NDArray& other);

	/*
	** Element-wise unary (return a new array; promote when the op requires it)
	*/
	NDArray neg() const;
	NDArray abs() const;
	NDArray sqrt() const;
	NDArray exp() const;
	NDArray log() const;
	NDArray floor() const;
	NDArray ceil() const;
	NDArray round() const;
	/** Unary minus; same as neg(). */
	NDArray operator-() const;

	/*
	** Element-wise binary (return a new array; same promote-on-op rules as + / *)
	*/
	NDArray minimum(const NDArray& other) const;
	NDArray maximum(const NDArray& other) const;
	NDArray pow(const NDArray& other) const;
	NDArray mod(const NDArray& other) const;

	NDArray minimum(float other) const;
	NDArray maximum(float other) const;
	NDArray pow(float other) const;
	NDArray mod(float other) const;

	NDArray minimum(int other) const;
	NDArray maximum(int other) const;
	NDArray pow(int other) const;
	NDArray mod(int other) const;

	/*
	** Element-wise comparisons → BINARY mask (same shape).
	** Whole-array equality remains operator== → bool.
	*/
	NDArray equal(const NDArray& other) const;
	NDArray notEqual(const NDArray& other) const;
	NDArray less(const NDArray& other) const;
	NDArray lessEqual(const NDArray& other) const;
	NDArray greater(const NDArray& other) const;
	NDArray greaterEqual(const NDArray& other) const;

	NDArray equal(float other) const;
	NDArray notEqual(float other) const;
	NDArray less(float other) const;
	NDArray lessEqual(float other) const;
	NDArray greater(float other) const;
	NDArray greaterEqual(float other) const;

	/**
	 * Element-wise select: result[i] = condition[i] ? x[i] : y[i].
	 * condition must be BINARY (or convertible); x and y are promoted to a
	 * common type. All three must share the same shape.
	 */
	static NDArray where(const NDArray& condition, const NDArray& x, const NDArray& y);

	/** True if any element is non-zero / true. */
	bool any() const;
	/** True if every element is non-zero / true. */
	bool all() const;

	/*
	** Reductions
	**
	** No-axis forms reduce to a scalar (empty shape).  Axis forms remove that
	** axis from the shape.  sum/prod promote integer types to UINT256 so
	** accumulation does not silently wrap; mean always returns F32.
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
	**    OPERATORS
	*/

	NDArray operator+(float other) const;
	NDArray operator-(float other) const;
	NDArray operator*(float other) const;
	NDArray operator/(float other) const;

	NDArray operator+(int other) const;
	NDArray operator-(int other) const;
	NDArray operator*(int other) const;
	NDArray operator/(int other) const;

	NDArray operator+(double other) const;
	NDArray operator-(double other) const;
	NDArray operator*(double other) const;
	NDArray operator/(double other) const;

	NDArray operator+(const NDArray& other) const;
	NDArray operator-(const NDArray& other) const;
	NDArray operator*(const NDArray& other) const;
	NDArray operator/(const NDArray& other) const;

	NDArray& operator+=(float other);
	NDArray& operator-=(float other);
	NDArray& operator*=(float other);
	NDArray& operator/=(float other);

	NDArray& operator+=(int other);
	NDArray& operator-=(int other);
	NDArray& operator*=(int other);
	NDArray& operator/=(int other);

	NDArray& operator+=(double other);
	NDArray& operator-=(double other);
	NDArray& operator*=(double other);
	NDArray& operator/=(double other);

	NDArray& operator+=(const NDArray& other);
	NDArray& operator-=(const NDArray& other);
	NDArray& operator*=(const NDArray& other);
	NDArray& operator/=(const NDArray& other);

	const ArrayList<int> shape;
	// Mutable so in-place arithmetic can promote without reallocating a new NDArray object.
	NDArrayType type;

private:
	enum class ArithOp { Add, Sub, Mul, Div };

	/** .cpp-only helpers that need access to storage pointers. */
	struct Impl;
	friend struct Impl;

	size_t initialize();
	size_t computeOffset(const ArrayList<int>& indices) const;

	float loadAsFloat(size_t i) const;
	uint256_t loadAsU256(size_t i) const;
	void storeFromFloat(size_t i, float v);
	void storeFromU256(size_t i, const uint256_t& v);

	/** Convert storage in place to `newType` (no-op if already that type). */
	void promoteInPlace(NDArrayType newType);

	void applyBinaryInPlace(const NDArray& src, ArithOp op);
	void applyFloatScalarInPlace(float scalar, ArithOp op);
	void applyIntScalarInPlace(int scalar, ArithOp op);

	/** Promote *this to a common type with `other`, then apply op. */
	NDArray& binaryOpInPlace(const NDArray& other, ArithOp op);
	/** Promote *this for a float/double scalar, then apply op. */
	NDArray& scalarFloatOpInPlace(float other, ArithOp op);
	/** Promote *this for an int scalar, then apply op. */
	NDArray& scalarIntOpInPlace(int other, ArithOp op);
	/**
	 * Broadcast `other` (shape prefix of *this) and apply op in place.
	 * Promotes *this to a common type with `other` first when needed.
	 */
	NDArray& broadcastOpInPlace(const NDArray& other, ArithOp op);
	/** Apply broadcast op; requires same type and prefix shape. */
	void applyBroadcastInPlace(const NDArray& src, ArithOp op);

	NDArray binaryOp(const NDArray& other, ArithOp op) const;
	NDArray scalarFloatOp(float other, ArithOp op) const;
	NDArray scalarIntOp(int other, ArithOp op) const;

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

	size_t memorySize;
	union {
		void* memory;

		uint8_t* uint8;

		// Used for uint64 *and* binary encoding
		uint64_t* uint64;

		float* float32;

		uint256_t* uint256;
	};
};


#endif //AGENT_CLUSTER_NDARRAY_H
