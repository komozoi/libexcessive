
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

	~NDArray();

	bool operator==(const NDArray& nds) const;

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
	 * Updates `*this`, so no copies or allocations are performed.
	 *
	 * @param other array to add
	 * @return *this
	 */
	NDArray& add(NDArray& other);

	/**
	 * Adds two NDArrays of different shapes, with `*this` being bigger and `other` being broadcasted to match the shape of `*this`.
	 *
	 * `other.shape` must be a prefix of `this->shape`
	 *
	 * Updates `*this`, so no copies or allocations are performed.
	 *
	 * @param other array to add
	 * @return *this
	 */
	NDArray& broadcastAdd(NDArray& other);

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
	const NDArrayType type;

private:
	size_t initialize();
	size_t computeOffset(const ArrayList<int>& indices) const;

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
