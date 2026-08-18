/*
 * Copyright 2021-2026 komozoi
 * Original Creation Date: 2026-7-15
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

#ifndef EXCESSIVE_NDARRAY_H
#define EXCESSIVE_NDARRAY_H

#include "bigint.h"
#include "stdint.h"
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "alloc/pointer.h"
#include "ds/ArrayList.h"


// Some types are commented out as not yet supported
enum NDArrayType {
	// Int types
	/** Packed 1-bit (64 values / uint64). */
	BINARY = 0x00,
	/** Signed 3-bit (-4..3), nibble-packed (16 values / uint64). */
	INT3 = 0x05,
	//INT4 = 0x06,
	//UINT4 = 0x07,
	INT8 = 0x0E,
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
	/** IEEE binary16 (1-5-10). Same-type arithmetic stays F16 (compute via F32). */
	F16 = 0x4A,
	/** bfloat16 (F32 exponent, 7-bit mantissa). Same-type arithmetic stays BF16. */
	BF16 = 0x4B,
	F32 = 0x4C,
	F64 = 0x4E,

	//BIGINT = 0x80
	/** 256-bit unsigned integer (four uint64 limbs). */
	UINT256 = 0x88
};

namespace ndarray_half {
	/** IEEE binary16 bits → F32. */
	inline float f16ToF32(uint16_t h) {
		const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
		const uint32_t exp = (h >> 10) & 0x1fu;
		uint32_t man = h & 0x3ffu;
		uint32_t bits;
		if (exp == 0) {
			if (man == 0) {
				bits = sign;
			} else {
				int32_t e = 127 - 14;
				while ((man & 0x400u) == 0) {
					man <<= 1;
					--e;
				}
				man &= 0x3ffu;
				bits = sign | ((uint32_t)e << 23) | (man << 13);
			}
		} else if (exp == 31) {
			bits = sign | 0x7f800000u | (man << 13);
		} else {
			bits = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
		}
		float f;
		std::memcpy(&f, &bits, 4);
		return f;
	}

	/** F32 → IEEE binary16 bits (round-to-nearest-even). */
	inline uint16_t f32ToF16(float f) {
		uint32_t bits;
		std::memcpy(&bits, &f, 4);
		const uint32_t sign = (bits >> 16) & 0x8000u;
		const int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
		const uint32_t man = bits & 0x7fffffu;
		if ((bits & 0x7fffffffu) > 0x7f800000u)
			return (uint16_t)(sign | 0x7e00u | (man >> 13));
		if (exp >= 31)
			return (uint16_t)(sign | 0x7c00u);
		if (exp <= 0) {
			if (exp < -10)
				return (uint16_t)sign;
			const uint32_t m = man | 0x800000u;
			const uint32_t shift = (uint32_t)(1 - exp);
			uint32_t half = m >> (13 + shift);
			const uint32_t rem = m & ((1u << (13 + shift)) - 1u);
			const uint32_t mid = 1u << (12 + shift);
			if (rem > mid || (rem == mid && (half & 1u)))
				++half;
			return (uint16_t)(sign | half);
		}
		uint32_t half = ((uint32_t)exp << 10) | (man >> 13);
		const uint32_t rem = man & 0x1fffu;
		if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
			++half;
		return (uint16_t)(sign | half);
	}

	/** bfloat16 bits → F32 (shift into the high half). */
	inline float bf16ToF32(uint16_t h) {
		const uint32_t bits = (uint32_t)h << 16;
		float f;
		std::memcpy(&f, &bits, 4);
		return f;
	}

	/** F32 → bfloat16 bits (round-to-nearest-even). */
	inline uint16_t f32ToBf16(float f) {
		uint32_t bits;
		std::memcpy(&bits, &f, 4);
		if ((bits & 0x7fffffffu) > 0x7f800000u)
			return (uint16_t)((bits >> 16) | 0x0040u);
		const uint32_t lsb = (bits >> 16) & 1u;
		bits += 0x7fffu + lsb;
		return (uint16_t)(bits >> 16);
	}

	/** Load F16 or BF16 bits as F32 (`t` must be F16 or BF16). */
	inline float load(NDArrayType t, uint16_t bits) {
		return t == F16 ? f16ToF32(bits) : bf16ToF32(bits);
	}
	/** Store F32 as F16 or BF16 bits (`t` must be F16 or BF16). */
	inline uint16_t store(NDArrayType t, float v) {
		return t == F16 ? f32ToF16(v) : f32ToBf16(v);
	}
}

/**
 * Shared contiguous buffer for NDArray / NDArrayView (managed via sp<> CoW).
 * Deep-copyable so COPY_ON_WRITE detach works.
 *
 * When ownsData is false the pointer is a wrap of caller-owned memory (mmap,
 * stack, JNI, …). The destructor must not free it. Copying the buffer always
 * allocates a new owned snapshot (used only if someone explicitly copy()s it);
 * wrap handles themselves never CoW-detach on write.
 */
struct NDArrayBuffer {
	void* data = nullptr;
	size_t byteSize = 0;
	NDArrayType elementType = F32;
	bool ownsData = true;

	/** Empty owned buffer (null data). */
	NDArrayBuffer() = default;
	/** Allocate `bytes` of zeroed owned storage. */
	NDArrayBuffer(size_t bytes, NDArrayType t);
	/** Non-owning wrap; does not free `external`. */
	NDArrayBuffer(void* external, size_t bytes, NDArrayType t);
	/** Deep copy: new owned snapshot of `other.data`. */
	NDArrayBuffer(const NDArrayBuffer& other);
	/** Move; `other` is left empty and non-owning. */
	NDArrayBuffer(NDArrayBuffer&& other) noexcept;
	/** Copy-assign: replace with an owned snapshot of `other`. */
	NDArrayBuffer& operator=(const NDArrayBuffer& other);
	/** Move-assign; `other` is left empty and non-owning. */
	NDArrayBuffer& operator=(NDArrayBuffer&& other) noexcept;
	/** Frees `data` only when `ownsData`. */
	~NDArrayBuffer();
};


class NDArray;
class NDArrayView;


/**
 * Non-owning (shared-buffer) view with independent shape/strides.
 * Survives destruction of the original NDArray via sp<> refcount.
 * Primarily for read + broadcast + materialise; in-place math materialises first.
 */
class NDArrayView {
public:

	/** Empty view: shape {0}, F32, null buffer. */
	NDArrayView() = default;

	/** View over `buf` with independent shape, element strides, and offset. */
	NDArrayView(sp<NDArrayBuffer> buf, ArrayList<int> shape, ArrayList<size_t> strides, size_t offset, NDArrayType type);

	/**
	 * View over caller-owned bytes (const or mutable). Does not copy or free
	 * `data`. The view has no mutators; writes go through `NDArray::wrap(void*)`.
	 * `data` must stay valid for the lifetime of this view and any copy of it.
	 * `byteSize` must be at least the packed size of `shape` × `type`.
	 * Axes must be >= 0; the packed size must fit in size_t.
	 * Packed types (BINARY, INT3, UINT256) require 8-byte alignment.
	 */
	static NDArrayView wrap(const void* data, size_t byteSize, ArrayList<int> shape, NDArrayType type);
	/**
	 * Same as wrap, with byte distance from row i to row i+1 (rank >= 2).
	 * Must be >= packed size of one row and aligned for `type`.
	 */
	static NDArrayView wrap(const void* data, size_t byteSize, ArrayList<int> shape, NDArrayType type,
	                        size_t rowStrideBytes);

	/** Product of shape (1 for a rank-0 scalar). Throws on a negative axis or overflow. */
	size_t numElements() const;
	/**
	 * Dense row-major unit strides. A nonzero element `offset` is allowed
	 * (contiguous row slice). Broadcast (stride 0) and gaps are not.
	 */
	bool isContiguous() const {
		size_t expect = 1;
		for (int d = shape.size() - 1; d >= 0; --d) {
			if (shape.get(d) <= 1)
				continue;
			if (d >= strides.size() || strides.get(d) != expect)
				return false;
			expect *= (size_t)shape.get(d);
		}
		return true;
	}
	/** Buffer element index of dense flat `i` (offset + i when contiguous). */
	size_t elementOffset(size_t flat) const {
		if (isContiguous())
			return offset + flat;
		size_t o = offset;
		size_t r = flat;
		for (int d = shape.size() - 1; d >= 0; --d) {
			int dim = shape.get(d);
			size_t c = (dim > 0) ? (r % (size_t)dim) : 0;
			if (dim > 0)
				r /= (size_t)dim;
			o += c * ((d < strides.size()) ? strides.get(d) : (size_t)0);
		}
		return o;
	}
	/** NumPy broadcast: right-align; each axis is equal or 1. */
	bool isBroadcastableTo(const ArrayList<int>& targetShape) const;
	/** View with `targetShape` and 0-stride on expanded axes. Throws if not broadcastable. */
	NDArrayView broadcastTo(const ArrayList<int>& targetShape) const;
	/** Contiguous reshape when product matches; otherwise throws. Keeps offset. */
	NDArrayView reshape(const ArrayList<int>& newShape) const;
	/**
	 * Owned reshape: shares the buffer when contiguous with offset 0,
	 * otherwise copy() then reshape. Product must match.
	 */
	NDArray reshapeOwned(const ArrayList<int>& newShape) const;
	/** Reverse axes (rank < 2 is a no-op). */
	NDArrayView transpose() const;
	/** Swap two axes. Throws if either axis is out of range. */
	NDArrayView swapaxes(int axisA, int axisB) const;
	/** Permute axes (must be a permutation of 0..rank-1). */
	NDArrayView permute(const ArrayList<int>& axes) const;
	/**
	 * Drop `axis` at `index`. `row(i)` is slice(0, i); `col(j)` is the last axis.
	 * Views share storage (read). Writes go through the owner or wrap().
	 */
	NDArrayView slice(int axis, int index) const;
	/** slice(0, i). */
	NDArrayView row(int i) const;
	/** slice(last axis, j). */
	NDArrayView col(int j) const;
	/** New owned NDArray with this view's logical elements (dense). */
	NDArray copy() const;

	/**
	 * Index of the first minimum / maximum. No-axis form is a scalar INT64
	 * flat index. Axis form removes that axis (index along it, 0..len-1).
	 */
	NDArray argmin() const;
	/** First minimum along `axis` (that axis is removed). */
	NDArray argmin(int axis) const;
	/** Flat INT64 index of the first maximum. */
	NDArray argmax() const;
	/** First maximum along `axis` (that axis is removed). */
	NDArray argmax(int axis) const;

	/**
	 * Widening scores. Accumulator type is the template argument — not a
	 * separate function per storage dtype. Products are widened into Acc
	 * (INT3 3·3 is 9, not wrap-mul 1). No intermediate length-n array.
	 *
	 * Pair ops require equal numElements() (flat pairing). Broadcast first
	 * if you need it. Contiguous same-type views take the packed fast path.
	 *
	 * Instantiated Acc: float, double, int32_t, int64_t, uint256_t.
	 */
	template <typename Acc>
	/** Widening inner product with `other`. */
	Acc dot(const NDArrayView& other) const;
	template <typename Acc>
	/** ||*this||² (widened). */
	Acc l2Squared() const;
	template <typename Acc>
	/** ||*this − other||² (widened diffs). */
	Acc l2Squared(const NDArrayView& other) const;
	template <typename Acc>
	/** sqrt(||*this||²) as Acc. */
	Acc l2Norm() const;
	template <typename Acc>
	/** sqrt(||*this − other||²) as Acc. */
	Acc l2Norm(const NDArrayView& other) const;
	template <typename Acc>
	/** Hamming distance. Both views must be BINARY. */
	Acc hamming(const NDArrayView& other) const;

	/**
	 * Full reductions with accumulator type as the template argument
	 * (same Acc set as dot, plus uint32_t / uint64_t). Packed contiguous
	 * views walk typed pointers / packed words; INT3 uses signed lanes
	 * (not wrap-mul). Empty throws. Defaults on NDArray::sum() keep the
	 * lossless NDArray policy.
	 */
	template <typename Acc>
	/** Sum of all elements into Acc. Empty throws. */
	Acc sumAs() const;
	template <typename Acc>
	/** Product of all elements into Acc. Empty throws. */
	Acc prodAs() const;
	template <typename Acc>
	/** mean = sumAs<Acc>() / numElements(). Empty throws. */
	Acc meanAs() const;

	/**
	 * A[...,M,K] @ B[...,K,N] → C[...,M,N]. Last two axes are the matrix;
	 * a leading axis is batch. Rank 3 is (B,M,K)@(B,K,N) → (B,M,N). A
	 * rank-2 operand broadcasts over the other's batch. Rank-1 is GEMV
	 * ((M,K)@(K,) → (M,), (K,)@(K,N) → (N,)); two rank-1 operands are an
	 * inner product (scalar). Both operands must be the same storage type.
	 * Result widens: BINARY / INT3 / INT8 / UINT8 → INT32, INT32 → INT64;
	 * INT64 / F32 / F64 / UINT256 stay. INT3 products are 3·3 = 9 (not
	 * wrap-mul). BINARY is popcount(A ∧ B). Type, rank, or inner-dim
	 * mismatch throws std::invalid_argument. Empty / zero-axis is empty.
	 */
	NDArray matmul(const NDArrayView& b) const;
	/** matmul against an NDArray (uses `b.view()`). */
	NDArray matmul(const NDArray& b) const;
	/** A[M,K] @ x[K] → y[M]. Same type and widening rules as matmul. */
	NDArray gemv(const NDArrayView& x) const;
	/** gemv against an NDArray (uses `x.view()`). */
	NDArray gemv(const NDArray& x) const;
	/**
	 * Group-scaled GEMV: y[i] = sum_g scale[i,g] * (W[i,g] · x[g]).
	 * W is [M,K], x is [K] (same type as W). groupSize > 0;
	 * nGroups = ceil(K / groupSize). scales is [M] when groupSize == K,
	 * else [M, nGroups]. Result is F32 [M]. UINT8 is 0..255 (no -128).
	 */
	NDArray gemv(const NDArrayView& x, const NDArrayView& scales, int groupSize) const;
	NDArray gemv(const NDArray& x, const NDArray& scales, int groupSize) const;

	template <typename T>
	/** Element at `indices` (rank must match). */
	T get(const ArrayList<int>& indices) const;
	template <typename T>
	/** Dense row-major flat index. Throws if `flat >= numElements()`. */
	T getFlat(size_t flat) const;

	/** Buffer element index of `indices` (shape rank; uses strides + offset). */
	size_t computeOffset(const ArrayList<int>& indices) const;

	/** Logical shape of this view. */
	const ArrayList<int>& getShape() const { return shape; }
	/** Element strides (0 = broadcast axis). */
	const ArrayList<size_t>& getStrides() const { return strides; }
	/** Element offset into the shared buffer. */
	size_t getOffset() const { return offset; }
	/** Storage type of the shared buffer. */
	NDArrayType getType() const { return type; }

	/** Shared storage (refcount); used by ndarray_detail load helpers. */
	const sp<NDArrayBuffer>& sharedBuffer() const { return buffer; }

	/** Packed word base of the shared buffer, or nullptr if empty. */
	const void* data() const {
		return buffer && buffer.get() ? buffer.get()->data : nullptr;
	}
	/** Size of the shared buffer in bytes (0 if empty). */
	size_t byteSize() const {
		return buffer && buffer.get() ? buffer.get()->byteSize : 0;
	}

private:
	ArrayList<int> shape;
	ArrayList<size_t> strides; // element strides; 0 = broadcast dimension
	size_t offset = 0;         // element offset into shared buffer
	NDArrayType type = F32;
	sp<NDArrayBuffer> buffer;
};


class NDArray {
public:
	/** Max rank for [] / initializer-list get/set; higher rank uses ArrayList overloads. */
	static constexpr int kMaxRank = 16;

	/** Mutable chained index proxy: a[i][j] = v / T(a[i][j]). Full rank required for R/W. */
	class Ref {
	public:
		/** Start a proxy at the first index. */
		Ref(NDArray* parent, int first);
		/** Append one index; read/write only after rank matches the array. */
		Ref operator[](int i) const;

		template <typename T>
		/** Load the element at the accumulated indices. */
		operator T() const;

		template <typename T>
		/** Store `value` at the accumulated indices. */
		Ref& operator=(const T& value);

	private:
		NDArray* parent;
		int idx[kMaxRank];
		int rank;
	};

	/** Const chained index proxy. */
	class CRef {
	public:
		/** Start a const proxy at the first index. */
		CRef(const NDArray* parent, int first);
		/** Append one index; convert to T after rank matches the array. */
		CRef operator[](int i) const;

		template <typename T>
		/** Load the element at the accumulated indices. */
		operator T() const;

	private:
		const NDArray* parent;
		int idx[kMaxRank];
		int rank;
	};

	/** Empty: shape {0}, type F32, no heap buffer. */
	NDArray();
	/** Dense owner of `shape` × `type` (zero-filled). A zero axis allocates nothing. */
	NDArray(ArrayList<int> shape, NDArrayType type);
	/** 1-D F32 from `vector`. */
	NDArray(const ArrayList<float>& vector);
	/** 1-D F64 from `vector`. */
	NDArray(const ArrayList<double>& vector);
	/** 1-D UINT8 from `vector`. */
	NDArray(const ArrayList<uint8_t>& vector);
	/** 1-D INT8 from `vector`. */
	NDArray(const ArrayList<int8_t>& vector);
	/** 1-D INT32 from `vector` (INT3 if every value is in {-1,0,1}). */
	NDArray(const ArrayList<int32_t>& vector);
	/** 1-D INT64 from `vector` (INT3 if every value is in {-1,0,1}). */
	NDArray(const ArrayList<int64_t>& vector);
	/** F32 owner of `shape` filled from `vector` (product must match). */
	NDArray(const ArrayList<int>& shape, const ArrayList<float>& vector);
	/** F64 / UINT8 / INT8 / INT32 / INT64 owner of `shape` from `vector`. */
	NDArray(const ArrayList<int>& shape, const ArrayList<double>& vector);
	/** UINT8 owner of `shape` filled from `vector`. */
	NDArray(const ArrayList<int>& shape, const ArrayList<uint8_t>& vector);
	/** INT8 owner of `shape` filled from `vector`. */
	NDArray(const ArrayList<int>& shape, const ArrayList<int8_t>& vector);
	/** INT32 owner of `shape` filled from `vector`. */
	NDArray(const ArrayList<int>& shape, const ArrayList<int32_t>& vector);
	/** INT64 owner of `shape` filled from `vector`. */
	NDArray(const ArrayList<int>& shape, const ArrayList<int64_t>& vector);

	template <typename T, typename = std::enable_if_t<std::is_same_v<T, NDArrayType>>>
	/** Rank-0 scalar of `type` holding `value`. */
	NDArray(T type, float value) : shape({}), type(static_cast<NDArrayType>(type)) {
		initialize();
		set({}, value);
	}

	template <typename T, typename = std::enable_if_t<std::is_same_v<T, NDArrayType>>>
	/** Rank-0 scalar of `type` holding `value`. */
	NDArray(T type, double value) : shape({}), type(static_cast<NDArrayType>(type)) {
		initialize();
		set({}, value);
	}

	template <typename T, typename = std::enable_if_t<std::is_same_v<T, NDArrayType>>>
	/** Rank-0 scalar of `type` holding `value`. */
	NDArray(T type, int value) : shape({}), type(static_cast<NDArrayType>(type)) {
		initialize();
		set({}, value);
	}

	/**
	 * Scalar array from an integer. Values -1, 0, 1 use INT3; other ints use INT32.
	 * Distinct from NDArray(NDArrayType, int) which always uses the given type.
	 */
	explicit NDArray(int value);
	/** Scalar from int64: INT3 for -1/0/1, otherwise INT64. */
	explicit NDArray(int64_t value);

	/**
	 * Wrap caller-owned bytes. Writes go to `data`; the array never CoW-detaches
	 * this buffer (that would orphan mmap). Caller keeps `data` alive for the
	 * lifetime of this array and any view/copy that still shares it.
	 * Copies of a wrap alias the same bytes (not an owned snapshot).
	 *
	 * Takes `void*`, not `const void*`: mutation is part of the NDArray API.
	 * Const memory is a view — use `NDArrayView::wrap`.
	 * Axes must be >= 0; the packed size must fit in size_t.
	 */
	static NDArray wrap(void* data, size_t byteSize, ArrayList<int> shape, NDArrayType type);
	/**
	 * Dense wrap when `rowStrideBytes` equals the packed row size.
	 * A padded stride throws; use NDArrayView::wrap.
	 */
	static NDArray wrap(void* data, size_t byteSize, ArrayList<int> shape, NDArrayType type,
	                    size_t rowStrideBytes);

	/** Empty 1-D array (shape {0}), no heap buffer. Type F32 if omitted. */
	static NDArray empty();
	/** Empty 1-D array (shape {0}), no heap buffer, given type. */
	static NDArray empty(NDArrayType type);

	/** Share the buffer (CoW). */
	NDArray(const NDArray& other);
	/** Move; source becomes a valid empty of the same type. */
	NDArray(NDArray&& other) noexcept;
	/** Releases the shared buffer (frees storage only if this was the last owner). */
	~NDArray();

	/** Share the buffer (CoW). */
	NDArray& operator=(const NDArray& other);
	/** Move-assign; source becomes a valid empty of the same type. */
	NDArray& operator=(NDArray&& other) noexcept;

	/**
	 * Whole-array equality (not element-wise).
	 * True only if both arrays have the same type, shape, and identical packed
	 * storage. This is a single bool for unit tests and structural checks.
	 * For an element-wise equality mask, use equal() or operator==(float/double/int).
	 */
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

	/** Number of logical elements (product of shape; 1 for a scalar).
	 *  Throws if an axis is negative or the product overflows size_t. */
	size_t numElements() const;

	/** True when numElements() == 0. Rank-0 scalar is not empty. */
	bool isEmpty() const;

	/** Row-major element strides for this dense array. */
	ArrayList<size_t> strides() const;
	/** Owners are packed row-major. */
	bool isContiguous() const { return true; }
	/** NumPy broadcast: right-align; each axis is equal or 1. */
	bool isBroadcastableTo(const ArrayList<int>& targetShape) const;
	/** Shared-buffer view of this owner (offset 0, dense strides). */
	NDArrayView view() const;
	/** view().broadcastTo(targetShape). */
	NDArrayView broadcastTo(const ArrayList<int>& targetShape) const;
	/** view().reshape(newShape); contiguous required. */
	NDArrayView reshapeView(const ArrayList<int>& newShape) const;
	/**
	 * Owned reshape: shares this buffer (CoW). Product must match.
	 * `[]` stays an element proxy; use row/slice for rank−1 views.
	 */
	NDArray reshape(const ArrayList<int>& newShape) const;
	/** view().transpose(). */
	NDArrayView transpose() const;
	/** view().swapaxes(axisA, axisB). */
	NDArrayView swapaxes(int axisA, int axisB) const;
	/** view().permute(axes). */
	NDArrayView permute(const ArrayList<int>& axes) const;
	/** view().slice(axis, index). */
	NDArrayView slice(int axis, int index) const;
	/** view().row(i). */
	NDArrayView row(int i) const;
	/** view().col(j). */
	NDArrayView col(int j) const;

	/** True if this array allocated (or CoW-copied) its storage. False for wrap(). */
	bool ownsStorage() const;

	/**
	 * Number of owned packed-buffer allocations (malloc of NDArrayBuffer data).
	 * Wraps do not increment. CoW snapshots do. For tests / instrumentation.
	 */
	static size_t ownedBufferAllocCount();

	/**
	 * Packed storage base, or nullptr if empty.
	 * Invalid after destroy, CoW detach, or any buffer replacement.
	 */
	const void* data() const {
		return buffer && buffer.get() ? buffer.get()->data : nullptr;
	}
	/** Mutable base. Calls ensureWritable() so CoW aliases are not mutated. */
	void* data() {
		ensureWritable();
		return buffer && buffer.get() ? buffer.get()->data : nullptr;
	}
	/** Size of the packed buffer in bytes (0 if empty). */
	size_t byteSize() const {
		return buffer && buffer.get() ? buffer.get()->byteSize : 0;
	}

	template <typename T>
	/** Typed packed base; throws if `T` does not match `type`. */
	const T* data() const;
	template <typename T>
	/** Mutable typed base (ensureWritable first). Throws on type mismatch. */
	T* data();

	template <typename Acc>
	/** Widening inner product; same contract as NDArrayView. */
	Acc dot(const NDArray& other) const { return view().dot<Acc>(other.view()); }
	template <typename Acc>
	/** Widening inner product with a view. */
	Acc dot(const NDArrayView& other) const { return view().dot<Acc>(other); }
	template <typename Acc>
	/** ||*this||² (widened). */
	Acc l2Squared() const { return view().l2Squared<Acc>(); }
	template <typename Acc>
	/** ||*this − other||² (widened). */
	Acc l2Squared(const NDArray& other) const { return view().l2Squared<Acc>(other.view()); }
	template <typename Acc>
	/** ||*this − other||² against a view. */
	Acc l2Squared(const NDArrayView& other) const { return view().l2Squared<Acc>(other); }
	template <typename Acc>
	/** sqrt(||*this||²) as Acc. */
	Acc l2Norm() const { return view().l2Norm<Acc>(); }
	template <typename Acc>
	/** sqrt(||*this − other||²) as Acc. */
	Acc l2Norm(const NDArray& other) const { return view().l2Norm<Acc>(other.view()); }
	template <typename Acc>
	/** sqrt(||*this − other||²) against a view. */
	Acc l2Norm(const NDArrayView& other) const { return view().l2Norm<Acc>(other); }
	template <typename Acc>
	/** Hamming distance. Both must be BINARY. */
	Acc hamming(const NDArray& other) const { return view().hamming<Acc>(other.view()); }
	template <typename Acc>
	/** Hamming distance against a view. */
	Acc hamming(const NDArrayView& other) const { return view().hamming<Acc>(other); }

	template <typename Acc>
	/** Sum of all elements into Acc. Empty throws. */
	Acc sumAs() const { return view().sumAs<Acc>(); }
	template <typename Acc>
	/** Product of all elements into Acc. Empty throws. */
	Acc prodAs() const { return view().prodAs<Acc>(); }
	template <typename Acc>
	/** mean = sumAs<Acc>() / numElements(). Empty throws. */
	Acc meanAs() const { return view().meanAs<Acc>(); }

	/**
	 * A[...,M,K] @ B[...,K,N] → C[...,M,N]. Last two axes are the matrix;
	 * a leading axis is batch. Rank 3 is (B,M,K)@(B,K,N) → (B,M,N). A
	 * rank-2 operand broadcasts over the other's batch. Rank-1 is GEMV
	 * ((M,K)@(K,) → (M,), (K,)@(K,N) → (N,)); two rank-1 operands are an
	 * inner product (scalar). Both operands must be the same storage type.
	 * Result widens: BINARY / INT3 / INT8 / UINT8 → INT32, INT32 → INT64;
	 * INT64 / F32 / F64 / UINT256 stay. INT3 products are 3·3 = 9 (not
	 * wrap-mul). BINARY is popcount(A ∧ B). Type, rank, or inner-dim
	 * mismatch throws std::invalid_argument. Empty / zero-axis is empty.
	 */
	NDArray matmul(const NDArray& b) const;
	/** matmul against a view. */
	NDArray matmul(const NDArrayView& b) const;
	/** A[M,K] @ x[K] → y[M]. Same type and widening rules as matmul. */
	NDArray gemv(const NDArray& x) const;
	/** gemv against a view. */
	NDArray gemv(const NDArrayView& x) const;
	/** Group-scaled GEMV. See NDArrayView::gemv(x, scales, groupSize). */
	NDArray gemv(const NDArray& x, const NDArray& scales, int groupSize) const;
	NDArray gemv(const NDArrayView& x, const NDArrayView& scales, int groupSize) const;

	/** Start a chained index proxy (`a[i][j]`). Full rank required to read/write. */
	Ref operator[](int i);
	/** Const chained index proxy. */
	CRef operator[](int i) const;

	template <typename T>
	/** Runtime multi-index (rank = shape.size()). */
	T get(const ArrayList<int>& indices) const {
		if (indices.size() != shape.size())
			throw std::out_of_range("NDArray::get - rank mismatch");
		return getAtOffset<T>(computeOffset(indices));
	}

	template <typename T>
	/** Store `value` at `indices` (rank must match). */
	void set(const ArrayList<int>& indices, const T& value) {
		if (indices.size() != shape.size())
			throw std::out_of_range("NDArray::set - rank mismatch");
		setAtOffset(computeOffset(indices), value);
	}

	template <typename T>
	/** get() with a brace list (`a.get<float>({i, j})`). Rank ≤ kMaxRank. */
	T get(std::initializer_list<int> indices) const {
		const int r = (int)indices.size();
		if (r > kMaxRank)
			throw std::out_of_range("NDArray::get - rank exceeds kMaxRank");
		int tmp[kMaxRank] = {};
		int n = 0;
		for (int x : indices)
			tmp[n++] = x;
		return getAtOffset<T>(computeOffset(tmp, n));
	}

	template <typename T>
	/** set() with a brace list. Rank ≤ kMaxRank. */
	void set(std::initializer_list<int> indices, const T& value) {
		const int r = (int)indices.size();
		if (r > kMaxRank)
			throw std::out_of_range("NDArray::set - rank exceeds kMaxRank");
		int tmp[kMaxRank] = {};
		int n = 0;
		for (int x : indices)
			tmp[n++] = x;
		setAtOffset(computeOffset(tmp, n), value);
	}

	template <typename T>
	/** Flat element index in dense row-major order. */
	T getFlat(size_t flat) const {
		if (flat >= numElements())
			throw std::out_of_range("NDArray::getFlat - index out of range");
		return getAtOffset<T>(flat);
	}

	template <typename T>
	/** Store `value` at dense row-major `flat`. */
	void setFlat(size_t flat, const T& value) {
		if (flat >= numElements())
			throw std::out_of_range("NDArray::setFlat - index out of range");
		setAtOffset(flat, value);
	}

	/*
	** Mathematical operations (operator overloads also present, these are the basic forms)
	**
	** add/sub/mul/div and += mutate *this (CoW-detach if the buffer is shared).
	** Operators and binaryOp allocate one result and do not convert() a side
	** that already has the promoted type. binaryOpInto writes a caller-owned dest.
	*/

	enum class ArithOp { Add, Sub, Mul, Div };

	/**
	 * dst[i] = a[i] ⊕ b[i]. dst must already have the promoted result type
	 * and the same shape as a and b. Does not allocate dst's buffer.
	 * Converts a and/or b only when that side is not already resultType.
	 */
	static void binaryOpInto(NDArray& dst, const NDArray& a, const NDArray& b, ArithOp op);

	/**
	 * Element-wise a ⊕ other into a new owned array (same shape).
	 * Same-type operands skip convert(); only a mismatched side is converted.
	 */
	NDArray binaryOp(const NDArray& other, ArithOp op) const;

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
	 * Adds `other` into `*this` with NumPy broadcasting.
	 *
	 * `other` must be broadcastable to `this->shape` (right-align; size-1
	 * axes expand). Destination shape is unchanged.
	 *
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 *
	 * @param other array to add
	 * @return *this
	 */
	NDArray& broadcastAdd(const NDArray& other);

	/**
	 * Subtracts a broadcasted `other` from `*this` (NumPy rule).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& broadcastSub(const NDArray& other);

	/**
	 * Multiplies `*this` by a broadcasted `other` (NumPy rule).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& broadcastMul(const NDArray& other);

	/**
	 * Divides `*this` by a broadcasted `other` (NumPy rule).
	 * Promotes `*this` when needed so precision is not lost.
	 * Does not mutate `other`.
	 */
	NDArray& broadcastDiv(const NDArray& other);

	/*
	** Factories (return new arrays)
	**
	** Defaults for constants -1 / 0 / 1 use INT3 (nibble-packed ternary).
	** Overloads that take NDArrayType keep that type (including *Like).
	*/
	/** Zeros, type INT3. */
	static NDArray zeros(const ArrayList<int>& shape);
	/** Zeros of the given type. */
	static NDArray zeros(const ArrayList<int>& shape, NDArrayType type);
	/** Ones, type INT3. */
	static NDArray ones(const ArrayList<int>& shape);
	/** Ones of the given type. */
	static NDArray ones(const ArrayList<int>& shape, NDArrayType type);
	/**
	 * Fill with an int constant. Type is INT3 when value is -1, 0, or 1;
	 * otherwise INT32.
	 */
	static NDArray full(const ArrayList<int>& shape, int value);
	/** Fill with an int64 constant (INT3 for -1/0/1, else INT64). */
	static NDArray full(const ArrayList<int>& shape, int64_t value);
	/** Fill `shape` with `value` stored as `type`. */
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, float value);
	/** Fill `shape` with a double stored as `type`. */
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, double value);
	/** Fill `shape` with an int stored as `type`. */
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, int value);
	/** Fill `shape` with an int64 stored as `type`. */
	static NDArray full(const ArrayList<int>& shape, NDArrayType type, int64_t value);
	/** Same shape/type as ref, all zeros. */
	static NDArray zerosLike(const NDArray& ref);
	/** Same shape/type as ref, all ones. */
	static NDArray onesLike(const NDArray& ref);
	/** Same shape/type as ref, every element `value`. */
	static NDArray fullLike(const NDArray& ref, float value);
	/** Same shape/type as ref, every element a double. */
	static NDArray fullLike(const NDArray& ref, double value);
	/** Same shape/type as ref, every element an int. */
	static NDArray fullLike(const NDArray& ref, int value);
	/** Same shape/type as ref, every element an int64. */
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
	/** In-place negate. UINT8 promotes to INT32. Returns *this. */
	NDArray& neg();
	/** In-place absolute value. Returns *this. */
	NDArray& abs();
	/** In-place sign: -1 / 0 / +1 (INT3: sign(-4) is -1). Returns *this. */
	NDArray& sign();
	/** In-place square. Returns *this. */
	NDArray& square();
	/** In-place sqrt (promotes to a real type). Returns *this. */
	NDArray& sqrt();
	/** In-place cube root. Returns *this. */
	NDArray& cbrt();
	/** In-place exp. Returns *this. */
	NDArray& exp();
	/** In-place expm1. Returns *this. */
	NDArray& expm1();
	/** In-place natural log. Returns *this. */
	NDArray& log();
	/** In-place log2. Returns *this. */
	NDArray& log2();
	/** In-place log10. Returns *this. */
	NDArray& log10();
	/** In-place log1p. Returns *this. */
	NDArray& log1p();
	/** In-place trig in radians; stay in the current float width. Returns *this. */
	NDArray& sin();
	/** In-place cosine. Returns *this. */
	NDArray& cos();
	/** In-place tangent. Returns *this. */
	NDArray& tan();
	/** In-place arcsine. Returns *this. */
	NDArray& asin();
	/** In-place arccosine. Returns *this. */
	NDArray& acos();
	/** In-place arctangent. Returns *this. */
	NDArray& atan();
	/** In-place sinh. Returns *this. */
	NDArray& sinh();
	/** In-place cosh. Returns *this. */
	NDArray& cosh();
	/** In-place tanh. Returns *this. */
	NDArray& tanh();
	/** In-place asinh. Returns *this. */
	NDArray& asinh();
	/** In-place acosh. Returns *this. */
	NDArray& acosh();
	/** In-place atanh. Returns *this. */
	NDArray& atanh();
	/** Degrees → radians (in place). */
	NDArray& deg2rad();
	/** Radians → degrees (in place). */
	NDArray& rad2deg();

	/** In-place floor / ceil / round. Returns *this. */
	NDArray& floor();
	/** In-place ceil. Returns *this. */
	NDArray& ceil();
	/** In-place round-to-nearest. Returns *this. */
	NDArray& round();

	/**
	 * In-place softmax over `axis` (default last). Stable: subtract the
	 * row max, exp, divide by the row sum. Integers promote like `exp()`.
	 * F16/BF16 stay half (compute in F32). Scalar → 1.
	 */
	NDArray& softmax();
	/** In-place softmax over `axis`. */
	NDArray& softmax(int axis);
	/** Copy then softmax (last axis). */
	NDArray softmaxed() const;
	/** Copy then softmax over `axis`. */
	NDArray softmaxed(int axis) const;

	/**
	 * RMSNorm over `axis` (default last):
	 *   out = x * weight / sqrt(mean(x²) + eps)
	 * `weight` is 1-D of length axis, or the same shape as *this.
	 * Result type follows *this after real-unary promotion.
	 */
	NDArray rmsnorm(const NDArray& weight, float eps = 1e-6f) const;
	/** RMSNorm over an explicit `axis`. */
	NDArray rmsnorm(const NDArray& weight, int axis, float eps) const;

	/**
	 * In-place SiLU: x / (1 + exp(-x)). Integers promote like `exp()`.
	 * `siluMul(other)` is fused `*this = silu(*this) * other`.
	 */
	NDArray& silu();
	/** Copy then SiLU. */
	NDArray silued() const;
	/** In-place fused SiLU(*this) * other. Returns *this. */
	NDArray& siluMul(const NDArray& other);

	/** Out-of-place: copy then square / neg / abs. */
	NDArray squared() const;
	/** Copy then negate. */
	NDArray negated() const;
	/** Copy then abs. */
	NDArray absolute() const;

	/**
	 * Two-argument arctangent: atan2(*this, x) element-wise (returns new array).
	 * Same convention as std::atan2(y, x) with *this as y.
	 */
	NDArray atan2(const NDArray& x) const;
	/** atan2 against a double scalar. */
	NDArray atan2(double x) const;
	/** atan2 against a float scalar. */
	NDArray atan2(float x) const;
	/** atan2 against an int scalar. */
	NDArray atan2(int x) const;

	/** Hypotenuse: hypot(*this, x) = sqrt(this² + x²), returns new array. */
	NDArray hypot(const NDArray& x) const;
	/** hypot against a double scalar. */
	NDArray hypot(double x) const;
	/** hypot against a float scalar. */
	NDArray hypot(float x) const;
	/** hypot against an int scalar. */
	NDArray hypot(int x) const;

	/** Unary minus — returns a new array (operators copy). */
	NDArray operator-() const;

	/*
	** Element-wise binary (return a new array; same promote-on-op rules as + / *)
	*/
	/** Element-wise min with `other` (same shape). */
	NDArray minimum(const NDArray& other) const;
	/** Element-wise max with `other` (same shape). */
	NDArray maximum(const NDArray& other) const;
	/** Element-wise power. Negative integer exponent throws. */
	NDArray pow(const NDArray& other) const;
	/** Element-wise remainder. Division by zero throws. */
	NDArray mod(const NDArray& other) const;
	/** Clamp each element into [lo, hi] (promotes as needed). */
	NDArray clip(const NDArray& lo, const NDArray& hi) const;
	/** Clamp into [lo, hi] with float bounds. */
	NDArray clip(float lo, float hi) const;
	/** Clamp into [lo, hi] with double bounds. */
	NDArray clip(double lo, double hi) const;

	/** Scalar overloads: broadcast `other` to every element. */
	NDArray minimum(float other) const;
	/** Element-wise max with a float scalar. */
	NDArray maximum(float other) const;
	/** Element-wise pow with a float scalar. */
	NDArray pow(float other) const;
	/** Element-wise mod with a float scalar. */
	NDArray mod(float other) const;

	/** Element-wise min with a double scalar. */
	NDArray minimum(double other) const;
	/** Element-wise max with a double scalar. */
	NDArray maximum(double other) const;
	/** Element-wise pow with a double scalar. */
	NDArray pow(double other) const;
	/** Element-wise mod with a double scalar. */
	NDArray mod(double other) const;

	/** Element-wise min with an int scalar. */
	NDArray minimum(int other) const;
	/** Element-wise max with an int scalar. */
	NDArray maximum(int other) const;
	/** Element-wise pow with an int scalar. */
	NDArray pow(int other) const;
	/** Element-wise mod with an int scalar. */
	NDArray mod(int other) const;

	/** Element-wise min with an int64 scalar. */
	NDArray minimum(int64_t other) const;
	/** Element-wise max with an int64 scalar. */
	NDArray maximum(int64_t other) const;
	/** Element-wise pow with an int64 scalar. */
	NDArray pow(int64_t other) const;
	/** Element-wise mod with an int64 scalar. */
	NDArray mod(int64_t other) const;

	/*
	** Element-wise comparisons
	**
	** These produce a new array with the same shape as *this (and other, when
	** both are arrays). They do not change *this.
	**
	** Two result types appear here:
	**   - NDArrayType::BINARY — packed 0/1 mask (false/true per element).
	**   - NDArrayType::INT3   — only from compare(): each element is -1, 0, or +1.
	**
	** Distinct from bool operator==(const NDArray&), which is whole-array
	** structural equality and returns a single C++ bool.
	**
	** Named methods (equal, less, …) and the matching operators (a == x with a
	** built-in number, a > b, …) are equivalent for the same operands. There is
	** no operator==(const NDArray&) that returns a mask, because that signature
	** is already used for whole-array bool equality.
	**
	** Masks feed select/where/choose, e.g. select(a > b, vb, va).
	*/

	/**
	 * Element-wise three-way comparison against another array of the same shape.
	 * @return NDArray of type INT3; each element is -1 if *this < other, 0 if
	 *         equal, +1 if *this > other (after the usual type promotion rules).
	 */
	NDArray compare(const NDArray& other) const;
	/**
	 * Element-wise three-way comparison against a C++ int broadcast to every element.
	 * @return INT3 array: -1 / 0 / +1 as in compare(const NDArray&).
	 */
	NDArray compare(int other) const;
	/**
	 * Element-wise three-way comparison against a float broadcast to every element.
	 * @return INT3 array: -1 / 0 / +1 as in compare(const NDArray&).
	 */
	NDArray compare(float other) const;
	/**
	 * Element-wise three-way comparison against a double broadcast to every element.
	 * @return INT3 array: -1 / 0 / +1 as in compare(const NDArray&).
	 */
	NDArray compare(double other) const;
	/**
	 * Element-wise three-way comparison against a uint256_t broadcast to every element.
	 * @return INT3 array: -1 / 0 / +1 as in compare(const NDArray&).
	 */
	NDArray compare(const uint256_t& other) const;

	/**
	 * Element-wise equality with another array of the same shape.
	 * @return BINARY mask: 1 where elements are equal, 0 elsewhere.
	 * @note For a single C++ bool of whole-array identity, use operator==(const NDArray&).
	 *       For equality against a number, use operator==(float|double|int).
	 */
	NDArray equal(const NDArray& other) const;
	/**
	 * Element-wise inequality with another array of the same shape.
	 * @return BINARY mask: 1 where elements differ, 0 where equal.
	 * Equivalent to operator!=(const NDArray&).
	 */
	NDArray notEqual(const NDArray& other) const;
	/**
	 * Element-wise less-than with another array of the same shape.
	 * @return BINARY mask: 1 where *this < other, else 0.
	 * Equivalent to operator<(const NDArray&).
	 */
	NDArray less(const NDArray& other) const;
	/**
	 * Element-wise less-or-equal with another array of the same shape.
	 * @return BINARY mask: 1 where *this <= other, else 0.
	 * Equivalent to operator<=(const NDArray&).
	 */
	NDArray lessEqual(const NDArray& other) const;
	/**
	 * Element-wise greater-than with another array of the same shape.
	 * @return BINARY mask: 1 where *this > other, else 0.
	 * Equivalent to operator>(const NDArray&).
	 */
	NDArray greater(const NDArray& other) const;
	/**
	 * Element-wise greater-or-equal with another array of the same shape.
	 * @return BINARY mask: 1 where *this >= other, else 0.
	 * Equivalent to operator>=(const NDArray&).
	 */
	NDArray greaterEqual(const NDArray& other) const;

	/**
	 * Element-wise greater-than vs another array (same shape).
	 * @return BINARY mask. Equivalent to greater(other).
	 */
	NDArray operator>(const NDArray& other) const;
	/**
	 * Element-wise less-than vs another array (same shape).
	 * @return BINARY mask. Equivalent to less(other).
	 */
	NDArray operator<(const NDArray& other) const;
	/**
	 * Element-wise greater-or-equal vs another array (same shape).
	 * @return BINARY mask. Equivalent to greaterEqual(other).
	 */
	NDArray operator>=(const NDArray& other) const;
	/**
	 * Element-wise less-or-equal vs another array (same shape).
	 * @return BINARY mask. Equivalent to lessEqual(other).
	 */
	NDArray operator<=(const NDArray& other) const;
	/**
	 * Element-wise inequality vs another array (same shape).
	 * @return BINARY mask. Equivalent to notEqual(other).
	 * @note This is not whole-array inequality; there is no bool operator!=(const NDArray&).
	 */
	NDArray operator!=(const NDArray& other) const;

	/**
	 * Element-wise equality of each entry of *this with a float value (broadcast).
	 * @return BINARY mask of the same shape as *this.
	 * Example: (vb == 0.0f) yields a mask true where vb is zero.
	 * @note operator==(const NDArray&) is a different overload and returns bool.
	 */
	NDArray operator==(float other) const;
	/** Element-wise inequality vs a float (broadcast). @return BINARY mask. */
	NDArray operator!=(float other) const;
	/** Element-wise greater-than vs a float (broadcast). @return BINARY mask. */
	NDArray operator>(float other) const;
	/** Element-wise less-than vs a float (broadcast). @return BINARY mask. */
	NDArray operator<(float other) const;
	/** Element-wise greater-or-equal vs a float (broadcast). @return BINARY mask. */
	NDArray operator>=(float other) const;
	/** Element-wise less-or-equal vs a float (broadcast). @return BINARY mask. */
	NDArray operator<=(float other) const;

	/** Element-wise equality vs a double (broadcast). @return BINARY mask. */
	NDArray operator==(double other) const;
	/** Element-wise inequality vs a double (broadcast). @return BINARY mask. */
	NDArray operator!=(double other) const;
	/** Element-wise greater-than vs a double (broadcast). @return BINARY mask. */
	NDArray operator>(double other) const;
	/** Element-wise less-than vs a double (broadcast). @return BINARY mask. */
	NDArray operator<(double other) const;
	/** Element-wise greater-or-equal vs a double (broadcast). @return BINARY mask. */
	NDArray operator>=(double other) const;
	/** Element-wise less-or-equal vs a double (broadcast). @return BINARY mask. */
	NDArray operator<=(double other) const;

	/** Element-wise equality vs an int (broadcast). @return BINARY mask. */
	NDArray operator==(int other) const;
	/** Element-wise inequality vs an int (broadcast). @return BINARY mask. */
	NDArray operator!=(int other) const;
	/** Element-wise greater-than vs an int (broadcast). @return BINARY mask. */
	NDArray operator>(int other) const;
	/** Element-wise less-than vs an int (broadcast). @return BINARY mask. */
	NDArray operator<(int other) const;
	/** Element-wise greater-or-equal vs an int (broadcast). @return BINARY mask. */
	NDArray operator>=(int other) const;
	/** Element-wise less-or-equal vs an int (broadcast). @return BINARY mask. */
	NDArray operator<=(int other) const;

	/**
	 * NDArray or scalar (int / float / double / int64_t) for value slots in
	 * select, choose, safeDiv, piecewise, etc. Scalars are written into the
	 * result only when chosen — no full temporary array is allocated for them.
	 */
	class ArrayOrScalar {
	public:
		/** Hold a reference to an array (not copied). */
		ArrayOrScalar(const NDArray& a);
		/** Hold a scalar (broadcast when selected). */
		ArrayOrScalar(double v);
		/** Hold a float scalar. */
		ArrayOrScalar(float v);
		/** Hold an int scalar. */
		ArrayOrScalar(int v);
		/** Hold an int64 scalar. */
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
	/** Two masks: first true wins, then `otherwise`. */
	static NDArray piecewise(const NDArray& m0, ArrayOrScalar v0,
	                         const NDArray& m1, ArrayOrScalar v1, ArrayOrScalar otherwise);
	/** Three masks: first true wins, then `otherwise`. */
	static NDArray piecewise(const NDArray& m0, ArrayOrScalar v0,
	                         const NDArray& m1, ArrayOrScalar v1,
	                         const NDArray& m2, ArrayOrScalar v2, ArrayOrScalar otherwise);

	/**
	 * Element-wise finiteness test → BINARY mask (same shape).
	 * Floats: std::isfinite. Integers / BINARY / INT3: always true.
	 * Use with any()/all(), e.g. a.isFinite().all().
	 */
	NDArray isFinite() const;
	/**
	 * Element-wise infinity test → BINARY mask (same shape).
	 * Floats: std::isinf. Integers / BINARY / INT3: always false.
	 * Use with any()/all(), e.g. a.isInfinite().any().
	 */
	NDArray isInfinite() const;
	/**
	 * Element-wise NaN test → BINARY mask (same shape).
	 * Floats: std::isnan. Integers / BINARY / INT3: always false.
	 * Use with any()/all(), e.g. a.isNan().any().
	 */
	NDArray isNaN() const;

	/*
	** BINARY / mask unary ops
	**
	** Word-wise on packed bits (not per-element RMW). Non-BINARY inputs are
	** first converted by truthiness (non-zero → 1) where noted.
	*/
	/** Truthiness → BINARY (share buffer if already BINARY). */
	NDArray asBinary() const;
	/** Logical NOT → new BINARY mask. Equivalent to ~(*this) after asBinary(). */
	NDArray logicalNot() const;
	/** Same as logicalNot(). */
	NDArray inverted() const;
	/** Bitwise / logical NOT on a BINARY view of *this. */
	NDArray operator~() const;
	/**
	 * In-place logical NOT. If not BINARY, replaces storage with asBinary() then
	 * inverts bits. Returns *this.
	 */
	NDArray& invert();

	/*
	** Mask / boolean ops
	**
	** logicalAnd / logicalOr are in-place boolean ops: both sides are reduced by
	** truthiness (non-zero → true), result is BINARY with only 0/1 bits. Safe when
	** values are multi-bit (e.g. INT3 holding 2): 2 | 1 → 1, not 3.
	** C++ has no &&= / ||=; use logicalAnd / logicalOr for that.
	**
	** & | ^ &= |= ^= operate on already-packed BINARY words (bit ops). Prefer
	** logicalAnd/logicalOr when operands may not be clean 0/1 masks.
	** && || ~ ! are out-of-place; &&/|| go through truthiness like logicalAnd/Or.
	*/
	/**
	 * In-place boolean AND. Always leaves *this as BINARY (even if it started as
	 * INT3/UINT8/F32/…); other is only read (truthiness). Element =
	 * truthy(*this) && truthy(other). Returns *this.
	 */
	NDArray& logicalAnd(const NDArray& other);
	/** In-place boolean AND with a scalar (false clears all; true normalizes to BINARY). */
	NDArray& logicalAnd(bool other);
	/**
	 * In-place boolean OR. Always leaves *this as BINARY; other is only read.
	 * Element = truthy(*this) || truthy(other). Returns *this.
	 */
	NDArray& logicalOr(const NDArray& other);
	/** In-place boolean OR with a scalar (true fills ones; false normalizes to BINARY). */
	NDArray& logicalOr(bool other);
	/**
	 * In-place boolean XOR. Always leaves *this as BINARY; other is only read.
	 * Element = truthy(*this) != truthy(other). Returns *this.
	 */
	NDArray& logicalXor(const NDArray& other);
	/** In-place boolean XOR with a scalar (true inverts mask; false normalizes to BINARY). */
	NDArray& logicalXor(bool other);

	/** Packed BINARY word AND (operands should already be 0/1 masks). */
	NDArray operator&(const NDArray& other) const;
	/** Packed BINARY word OR. */
	NDArray operator|(const NDArray& other) const;
	/** Packed BINARY word XOR. */
	NDArray operator^(const NDArray& other) const;
	/** Out-of-place boolean AND (truthiness → BINARY, then word AND). */
	NDArray operator&&(const NDArray& other) const;
	/** Out-of-place boolean OR (truthiness → BINARY, then word OR). */
	NDArray operator||(const NDArray& other) const;
	/** Out-of-place boolean NOT. */
	NDArray operator!() const;

	/** In-place packed BINARY AND. Returns *this. */
	NDArray& operator&=(const NDArray& other);
	/** In-place packed BINARY OR. Returns *this. */
	NDArray& operator|=(const NDArray& other);
	/** In-place packed BINARY XOR. Returns *this. */
	NDArray& operator^=(const NDArray& other);

	/** True if any element is non-zero / true. */
	bool any() const;
	/** True if every element is non-zero / true. */
	bool all() const;
	/** counts the number of nonzero elements */
	int countNonzero() const;

	/*
	** Reductions
	**
	** No-axis forms reduce to a scalar (empty shape).  Axis forms remove that
	** axis from the shape.  BINARY / INT3 / UINT8 / INT8 sum/prod use INT32;
	** INT32 uses INT64; INT64 and UINT256 stay; floats keep their type.
	** mean uses F64 except pure F32 inputs (stay F32).
	**
	** sumAs<Acc>() / prodAs<Acc>() / meanAs<Acc>() request a cheap machine
	** accumulator (see NDArrayView). Defaults above do not change.
	*/
	/** Full reduction to a rank-0 scalar (lossless default type). Empty throws. */
	NDArray sum() const;
	/** Reduce along `axis` (that axis is removed). Empty throws. */
	NDArray sum(int axis) const;
	/** Mean of all elements (rank-0). Empty throws. */
	NDArray mean() const;
	/** Mean along `axis`. Empty throws. */
	NDArray mean(int axis) const;
	/** Minimum of all elements (rank-0). Empty throws. */
	NDArray min() const;
	/** Minimum along `axis`. Empty throws. */
	NDArray min(int axis) const;
	/** Maximum of all elements (rank-0). Empty throws. */
	NDArray max() const;
	/** Maximum along `axis`. Empty throws. */
	NDArray max(int axis) const;
	/** Product of all elements (rank-0). Empty throws. */
	NDArray prod() const;
	/** Product along `axis`. Empty throws. */
	NDArray prod(int axis) const;
	/**
	 * Index of the first min / max. Scalar INT64 (flat) or INT64 with `axis`
	 * removed. Same convention on every dtype (INT3 uses signed lanes).
	 */
	NDArray argmin() const;
	/** First minimum along `axis`. */
	NDArray argmin(int axis) const;
	/** Flat INT64 index of the first maximum. */
	NDArray argmax() const;
	/** First maximum along `axis`. */
	NDArray argmax(int axis) const;

	/*
	**    OPERATORS  (return new arrays — they copy)
	*/

	/** Element-wise + with a float scalar. */
	NDArray operator+(float other) const;
	/** Element-wise − with a float scalar. */
	NDArray operator-(float other) const;
	/** Element-wise * with a float scalar. */
	NDArray operator*(float other) const;
	/** Element-wise / with a float scalar. */
	NDArray operator/(float other) const;
	/** Element-wise % with a float scalar. */
	NDArray operator%(float other) const;

	/** Element-wise + with an int scalar. */
	NDArray operator+(int other) const;
	/** Element-wise − with an int scalar. */
	NDArray operator-(int other) const;
	/** Element-wise * with an int scalar. */
	NDArray operator*(int other) const;
	/** Element-wise / with an int scalar. */
	NDArray operator/(int other) const;
	/** Element-wise % with an int scalar. */
	NDArray operator%(int other) const;

	/** Element-wise + with an int64 scalar. */
	NDArray operator+(int64_t other) const;
	/** Element-wise − with an int64 scalar. */
	NDArray operator-(int64_t other) const;
	/** Element-wise * with an int64 scalar. */
	NDArray operator*(int64_t other) const;
	/** Element-wise / with an int64 scalar. */
	NDArray operator/(int64_t other) const;
	/** Element-wise % with an int64 scalar. */
	NDArray operator%(int64_t other) const;

	/** Element-wise + with a double scalar. */
	NDArray operator+(double other) const;
	/** Element-wise − with a double scalar. */
	NDArray operator-(double other) const;
	/** Element-wise * with a double scalar. */
	NDArray operator*(double other) const;
	/** Element-wise / with a double scalar. */
	NDArray operator/(double other) const;
	/** Element-wise % with a double scalar. */
	NDArray operator%(double other) const;

	/** Element-wise + with another array (same shape). */
	NDArray operator+(const NDArray& other) const;
	/** Element-wise − with another array. */
	NDArray operator-(const NDArray& other) const;
	/** Element-wise * with another array. */
	NDArray operator*(const NDArray& other) const;
	/** Element-wise / with another array. */
	NDArray operator/(const NDArray& other) const;
	/** Element-wise modulo; same as mod(). */
	NDArray operator%(const NDArray& other) const;

	/** In-place += float. Returns *this. */
	NDArray& operator+=(float other);
	/** In-place -= float. Returns *this. */
	NDArray& operator-=(float other);
	/** In-place *= float. Returns *this. */
	NDArray& operator*=(float other);
	/** In-place /= float. Returns *this. */
	NDArray& operator/=(float other);
	/** In-place %= float. Returns *this. */
	NDArray& operator%=(float other);

	/** In-place += int. Returns *this. */
	NDArray& operator+=(int other);
	/** In-place -= int. Returns *this. */
	NDArray& operator-=(int other);
	/** In-place *= int. Returns *this. */
	NDArray& operator*=(int other);
	/** In-place /= int. Returns *this. */
	NDArray& operator/=(int other);
	/** In-place %= int. Returns *this. */
	NDArray& operator%=(int other);

	/** In-place += int64. Returns *this. */
	NDArray& operator+=(int64_t other);
	/** In-place -= int64. Returns *this. */
	NDArray& operator-=(int64_t other);
	/** In-place *= int64. Returns *this. */
	NDArray& operator*=(int64_t other);
	/** In-place /= int64. Returns *this. */
	NDArray& operator/=(int64_t other);
	/** In-place %= int64. Returns *this. */
	NDArray& operator%=(int64_t other);

	/** In-place += double. Returns *this. */
	NDArray& operator+=(double other);
	/** In-place -= double. Returns *this. */
	NDArray& operator-=(double other);
	/** In-place *= double. Returns *this. */
	NDArray& operator*=(double other);
	/** In-place /= double. Returns *this. */
	NDArray& operator/=(double other);
	/** In-place %= double. Returns *this. */
	NDArray& operator%=(double other);

	/** In-place += array. Returns *this. */
	NDArray& operator+=(const NDArray& other);
	/** In-place -= array. Returns *this. */
	NDArray& operator-=(const NDArray& other);
	/** In-place *= array. Returns *this. */
	NDArray& operator*=(const NDArray& other);
	/** In-place /= array. Returns *this. */
	NDArray& operator/=(const NDArray& other);
	/** In-place %= array. Returns *this. */
	NDArray& operator%=(const NDArray& other);

	ArrayList<int> shape;
	// Mutable so in-place arithmetic can promote; kept consistent with buffer dtype.
	NDArrayType type;

private:
	/** .cpp-only helpers that need access to storage pointers. */
	struct Impl;
	friend struct Impl;
	friend class NDArrayView;

	/** Allocate / bind storage for the current shape and type. */
	size_t initialize();
	/** Dense row-major offset of `indices` (rank = shape.size()). */
	size_t computeOffset(const ArrayList<int>& indices) const;
	/** Dense offset from a C array of `rank` indices. */
	size_t computeOffset(const int* indices, int rank) const;
	/** Refresh typed pointers from `buffer`. */
	void rebindPointers();
	/** CoW-detach owned buffers that are shared; wraps write through. */
	void ensureWritable();
	/** Row-major element strides for `shape`. */
	static ArrayList<size_t> rowMajorStrides(const ArrayList<int>& shape);
	/** Packed byte size of `numElems` of type `t`. */
	static size_t bufferBytesFor(NDArrayType t, size_t numElems);
	/** Mutable wrap buffer (UNIQUE, write-through). */
	static sp<NDArrayBuffer> makeWrapBuffer(void* data, size_t byteSize, const ArrayList<int>& shape, NDArrayType type);
	/** Const wrap: COPY_ON_WRITE handle (mutation should detach). */
	static sp<NDArrayBuffer> makeWrapBuffer(const void* data, size_t byteSize, const ArrayList<int>& shape, NDArrayType type);

	/** Adopt an existing buffer (used by reshape / reshapeOwned). */
	NDArray(ArrayList<int> shape, NDArrayType type, sp<NDArrayBuffer> buf);

	template <typename T>
	/** Load the element at packed index `offset` as T. */
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
			case INT8:
				if constexpr (std::is_same_v<T, uint256_t>)
					return uint256_t((int)int8[offset]);
				else
					return static_cast<T>(int8[offset]);
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
			case F16:
			case BF16:
				return static_cast<T>(ndarray_half::load(type, u16[offset]));
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

	/** BINARY store: 1 iff value > 0 (floats / signed ints). bool is itself.
	 *  Matches storeFromDouble / storeFromI64. bool is special-cased so
	 *  `value > 0` never instantiates under -Werror=bool-compare. */
	template <typename T>
	/** BINARY store: 1 iff value is truthy (see comment above). */
	static bool toBinaryBit(const T& value) {
		if constexpr (std::is_same_v<T, bool>)
			return value;
		else if constexpr (std::is_floating_point_v<T>)
			return value > static_cast<T>(0);
		else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
			return value > 0;
		else if constexpr (std::is_integral_v<T>)
			return value != 0;
		else
			return static_cast<bool>(value);
	}

	template <typename T>
	/** Store `value` at packed index `offset`. */
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
				case INT8:
					int8[offset] = (int8_t)(uint64_t)value;
					break;
				case INT32:
					int32[offset] = (int32_t)(uint64_t)value;
					break;
				case INT64:
					int64[offset] = (int64_t)(uint64_t)value;
					break;
				case F16:
				case BF16:
					u16[offset] = ndarray_half::store(type, (float)value.toDouble());
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
					if (toBinaryBit(value))
						uint64[offset >> 6] |= 1ULL << (offset & 63);
					else
						uint64[offset >> 6] &= ~(1ULL << (offset & 63));
					break;
				case INT3: {
					int iv = static_cast<int>(value);
					uint8_t p = (uint8_t)(iv & 7);
					const size_t w = offset / 16;
					const unsigned sh = (unsigned)((offset % 16) * 4);
					uint64[w] = (uint64[w] & ~(0xFULL << sh)) | ((uint64_t)p << sh);
					break;
				}
				case UINT8:
					uint8[offset] = static_cast<uint8_t>(value);
					break;
				case INT8:
					int8[offset] = static_cast<int8_t>(value);
					break;
				case INT32:
					int32[offset] = static_cast<int32_t>(value);
					break;
				case INT64:
					int64[offset] = static_cast<int64_t>(value);
					break;
				case F16:
				case BF16:
					u16[offset] = ndarray_half::store(type, static_cast<float>(value));
					break;
				case F32:
					float32[offset] = static_cast<float>(value);
					break;
				case F64:
					float64[offset] = static_cast<double>(value);
					break;
				case UINT256:
					// bool is integral but must not use `value < 0` (-Werror=bool-compare).
					if constexpr (std::is_same_v<T, bool>)
						uint256[offset] = value ? uint256_t(1) : uint256_t(0);
					else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
						uint256[offset] = value < 0
							? uint256_t((int)value)
							: uint256_t((uint64_t)value);
					else if constexpr (std::is_integral_v<T>)
						uint256[offset] = uint256_t((uint64_t)value);
					else if constexpr (std::is_floating_point_v<T>)
						uint256[offset] = uint256_t((double)value);
					else
						uint256[offset] = uint256_t((uint64_t)0);
					break;
			}
		}
	}

	/** Load packed index `i` widened to float / double / int64 / uint256. */
	float loadAsFloat(size_t i) const;
	/** Load packed index `i` as double. */
	double loadAsDouble(size_t i) const;
	/** Load packed index `i` as int64. */
	int64_t loadAsI64(size_t i) const;
	/** Load packed index `i` as uint256. */
	uint256_t loadAsU256(size_t i) const;
	/** Store into packed index `i` (ensureWritable first). */
	void storeFromFloat(size_t i, float v);
	/** Store a double into packed index `i`. */
	void storeFromDouble(size_t i, double v);
	/** Store an int64 into packed index `i`. */
	void storeFromI64(size_t i, int64_t v);
	/** Store a uint256 into packed index `i`. */
	void storeFromU256(size_t i, const uint256_t& v);

	/** Convert storage in place to `newType` (no-op if already that type). */
	void promoteInPlace(NDArrayType newType);

	/** *this[i] ⊕= src[i]. Same type and shape. */
	void applyBinaryInPlace(const NDArray& src, ArithOp op);
	/** *this[i] = a[i] ⊕ b[i]. a and b must already match this->type. */
	void applyBinaryInto(const NDArray& a, const NDArray& b, ArithOp op);
	/** Apply `op` with a scalar in place. */
	void applyFloatScalarInPlace(float scalar, ArithOp op);
	/** Apply `op` with a double scalar in place. */
	void applyDoubleScalarInPlace(double scalar, ArithOp op);
	/** Apply `op` with an int scalar in place. */
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
	/** Apply `op` with an int64 scalar in place. */
	void applyInt64ScalarInPlace(int64_t scalar, ArithOp op);
	/**
	 * Broadcast `other` (shape prefix of *this) and apply op in place.
	 * Promotes *this to a common type with `other` first when needed.
	 */
	NDArray& broadcastOpInPlace(const NDArray& other, ArithOp op);
	/** Apply broadcast op; requires same type and prefix shape. */
	void applyBroadcastInPlace(const NDArrayView& src, ArithOp op);

	/** Out-of-place scalar ⊕ *this (copy, then apply). */
	NDArray scalarFloatOp(float other, ArithOp op) const;
	/** Out-of-place double scalar ⊕ *this. */
	NDArray scalarDoubleOp(double other, ArithOp op) const;
	/** Out-of-place int scalar ⊕ *this. */
	NDArray scalarIntOp(int other, ArithOp op) const;
	/** Out-of-place int64 scalar ⊕ *this. */
	NDArray scalarInt64Op(int64_t other, ArithOp op) const;

	/** Promote in place to a real type and apply unary kernel (F32 or F64). */
	NDArray& mapRealUnary(double (*fn)(double));

	/** Replace storage with that of `result` (moves buffer; used by %= etc.). */
	void stealFrom(NDArray& result);

	/** Fill every element with value (cast into current type). */
	void fillFromDouble(double value);
	/** Fill every element from an int64 (cast into current type). */
	void fillFromI64(int64_t value);

	template <typename T>
	/** Cast a uint256 limb value to T. */
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
		int8_t* int8;
		int32_t* int32;
		int64_t* int64;

		// Used for uint64 *and* binary encoding
		uint64_t* uint64;

		uint16_t* u16;
		float* float32;
		double* float64;

		uint256_t* uint256;
	};
};

using NDArrayRef = NDArray::Ref;
using NDArrayCRef = NDArray::CRef;


template <typename T>
/** True if packed `data<T>()` is valid for storage type `t`. */
static bool ndarrayDataTypeMatches(NDArrayType t) {
	if constexpr (std::is_same_v<T, float>)
		return t == F32;
	else if constexpr (std::is_same_v<T, double>)
		return t == F64;
	else if constexpr (std::is_same_v<T, uint8_t>)
		return t == UINT8;
	else if constexpr (std::is_same_v<T, int8_t>)
		return t == INT8;
	else if constexpr (std::is_same_v<T, int32_t>)
		return t == INT32;
	else if constexpr (std::is_same_v<T, int64_t>)
		return t == INT64;
	else if constexpr (std::is_same_v<T, uint256_t>)
		return t == UINT256;
	else if constexpr (std::is_same_v<T, uint16_t>)
		return t == F16 || t == BF16;
	else if constexpr (std::is_same_v<T, uint64_t>)
		return t == BINARY || t == INT3;
	else
		return false;
}

template <typename T>
/** Typed const packed base; throws on type mismatch. */
const T* NDArray::data() const {
	if (!ndarrayDataTypeMatches<T>(type))
		throw std::invalid_argument("NDArray::data<T>: type mismatch");
	return static_cast<const T*>(data());
}

template <typename T>
/** Typed mutable packed base; throws on type mismatch. */
T* NDArray::data() {
	if (!ndarrayDataTypeMatches<T>(type))
		throw std::invalid_argument("NDArray::data<T>: type mismatch");
	return static_cast<T*>(data());
}

// ---- proxy template methods (need NDArray complete) -------------------------

template <typename T>
/** Load the element at the proxy indices. */
NDArray::Ref::operator T() const {
	return parent->getAtOffset<T>(parent->computeOffset(idx, rank));
}

template <typename T>
/** Store `value` at the proxy indices. */
NDArray::Ref& NDArray::Ref::operator=(const T& value) {
	parent->setAtOffset(parent->computeOffset(idx, rank), value);
	return *this;
}

template <typename T>
/** Load the element at the const proxy indices. */
NDArray::CRef::operator T() const {
	return parent->getAtOffset<T>(parent->computeOffset(idx, rank));
}

// View element access: defined in NDArray.cpp (uses shared buffer load helpers)
// Explicit instantiations provided for common T; generic path uses convert via double/i64.
namespace ndarray_detail {
	/** Load one view element as double / int64 / uint256. */
	double viewLoadDouble(const NDArrayView& v, size_t elementOffset);
	/** Load one view element as int64 (typed pointer / packed word). */
	int64_t viewLoadI64(const NDArrayView& v, size_t elementOffset);
	/** Load one view element as uint256. */
	uint256_t viewLoadU256(const NDArrayView& v, size_t elementOffset);
}

template <typename T>
/** Dense row-major flat index. Throws if `flat >= numElements()`. */
T NDArrayView::getFlat(size_t flat) const {
	if (flat >= numElements())
		throw std::out_of_range("NDArrayView::getFlat - index out of range");
	const size_t elemOff = elementOffset(flat);
	if constexpr (std::is_same_v<T, uint256_t>)
		return ndarray_detail::viewLoadU256(*this, elemOff);
	else if constexpr (std::is_floating_point_v<T>)
		return static_cast<T>(ndarray_detail::viewLoadDouble(*this, elemOff));
	else
		return static_cast<T>(ndarray_detail::viewLoadI64(*this, elemOff));
}

template <typename T>
/** Element at `indices` (rank must match). */
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


/** Left-hand float + NDArray. */
NDArray operator+(float lhs, const NDArray& rhs);
/** Left-hand float − NDArray. */
NDArray operator-(float lhs, const NDArray& rhs);
/** Left-hand float * NDArray. */
NDArray operator*(float lhs, const NDArray& rhs);
/** Left-hand float / NDArray. */
NDArray operator/(float lhs, const NDArray& rhs);
/** Left-hand float % NDArray. */
NDArray operator%(float lhs, const NDArray& rhs);

/** Left-hand double + NDArray. */
NDArray operator+(double lhs, const NDArray& rhs);
/** Left-hand double − NDArray. */
NDArray operator-(double lhs, const NDArray& rhs);
/** Left-hand double * NDArray. */
NDArray operator*(double lhs, const NDArray& rhs);
/** Left-hand double / NDArray. */
NDArray operator/(double lhs, const NDArray& rhs);
/** Left-hand double % NDArray. */
NDArray operator%(double lhs, const NDArray& rhs);

/** Left-hand int + NDArray. */
NDArray operator+(int lhs, const NDArray& rhs);
/** Left-hand int − NDArray. */
NDArray operator-(int lhs, const NDArray& rhs);
/** Left-hand int * NDArray. */
NDArray operator*(int lhs, const NDArray& rhs);
/** Left-hand int / NDArray. */
NDArray operator/(int lhs, const NDArray& rhs);
/** Left-hand int % NDArray. */
NDArray operator%(int lhs, const NDArray& rhs);

/** Left-hand int64 + NDArray. */
NDArray operator+(int64_t lhs, const NDArray& rhs);
/** Left-hand int64 − NDArray. */
NDArray operator-(int64_t lhs, const NDArray& rhs);
/** Left-hand int64 * NDArray. */
NDArray operator*(int64_t lhs, const NDArray& rhs);
/** Left-hand int64 / NDArray. */
NDArray operator/(int64_t lhs, const NDArray& rhs);
/** Left-hand int64 % NDArray. */
NDArray operator%(int64_t lhs, const NDArray& rhs);

/**
 * Left-hand number vs NDArray comparisons (element-wise).
 * Each returns a BINARY mask with the same shape as rhs.
 */
NDArray operator==(float lhs, const NDArray& rhs);
/** Element-wise: float != each element of rhs. @return BINARY mask. */
NDArray operator!=(float lhs, const NDArray& rhs);
/** Element-wise: float > each element of rhs. @return BINARY mask. */
NDArray operator>(float lhs, const NDArray& rhs);
/** Element-wise: float < each element of rhs. @return BINARY mask. */
NDArray operator<(float lhs, const NDArray& rhs);
/** Element-wise: float >= each element of rhs. @return BINARY mask. */
NDArray operator>=(float lhs, const NDArray& rhs);
/** Element-wise: float <= each element of rhs. @return BINARY mask. */
NDArray operator<=(float lhs, const NDArray& rhs);

/** Element-wise: double == each element of rhs. @return BINARY mask. */
NDArray operator==(double lhs, const NDArray& rhs);
/** Element-wise: double != each element of rhs. @return BINARY mask. */
NDArray operator!=(double lhs, const NDArray& rhs);
/** Element-wise: double > each element of rhs. @return BINARY mask. */
NDArray operator>(double lhs, const NDArray& rhs);
/** Element-wise: double < each element of rhs. @return BINARY mask. */
NDArray operator<(double lhs, const NDArray& rhs);
/** Element-wise: double >= each element of rhs. @return BINARY mask. */
NDArray operator>=(double lhs, const NDArray& rhs);
/** Element-wise: double <= each element of rhs. @return BINARY mask. */
NDArray operator<=(double lhs, const NDArray& rhs);

/** Element-wise: int == each element of rhs. @return BINARY mask. */
NDArray operator==(int lhs, const NDArray& rhs);
/** Element-wise: int != each element of rhs. @return BINARY mask. */
NDArray operator!=(int lhs, const NDArray& rhs);
/** Element-wise: int > each element of rhs. @return BINARY mask. */
NDArray operator>(int lhs, const NDArray& rhs);
/** Element-wise: int < each element of rhs. @return BINARY mask. */
NDArray operator<(int lhs, const NDArray& rhs);
/** Element-wise: int >= each element of rhs. @return BINARY mask. */
NDArray operator>=(int lhs, const NDArray& rhs);
/** Element-wise: int <= each element of rhs. @return BINARY mask. */
NDArray operator<=(int lhs, const NDArray& rhs);


#endif // EXCESSIVE_NDARRAY_H
