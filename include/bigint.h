/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-07-14
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


#ifndef EXCESSIVE_BIGINT_H
#define EXCESSIVE_BIGINT_H

#include "stdint.h"
#include "LongKey.h"
#include "moremath.h"


void _internalBigintFastAdd(int aSize, const uint64_t* a, int bSize, const uint64_t* b, int ySize, uint64_t* y);
void _internalBigintFastSub(int aSize, const uint64_t* a, int bSize, const uint64_t* b, int ySize, uint64_t* y);
void _internalBigintFastMul(int size, const uint64_t* a, const uint64_t* b, uint64_t* y);
void _internalBigintFastDiv(uint64_t* remainder, const uint64_t* divisor, uint64_t* quotient, int size);

void _internalBigintFastPow(const uint64_t* x, uint64_t* y, int size, int power);
void _internalBigintFastRoot(const uint64_t* x, uint64_t* y, int size, int root);
void _internalBigintFastSqrt(const uint64_t* x, uint64_t* y, int size);
void _internalBigintFastSmallRoot(const uint64_t* x, uint64_t* y, int size, int root);

double _internalBigintConvertToDouble(uint16_t bits, int N, const uint64_t* chunks);
void _internalBigintConvertFromDouble(double value, int N, uint64_t* chunks);
void _internalBigintFastMulDouble(const uint64_t* a, double b, uint64_t* y, int size);


/**
 * @brief A fixed-width unsigned big integer class.
 * 
 * Inherits from LongKey for basic storage and comparison, adding arithmetic operations.
 * 
 * @tparam N The number of 64-bit chunks used to store the integer.
 */
template<int N>
class UnsignedFixedWidthBigInt: public LongKey<N*64> {
public:
	/**
	 * @brief Default constructor. Initializes to zero.
	 */
	UnsignedFixedWidthBigInt() = default;

	/**
	 * @brief Constructs from a byte buffer.
	 * @param buffer Pointer to the byte buffer.
	 * @param isBigEndian True if the buffer is in big-endian format.
	 */
	UnsignedFixedWidthBigInt(const uint8_t* buffer, bool isBigEndian)
		: LongKey<N * 64>(buffer, isBigEndian) {}

	/**
	 * @brief Constructs from a byte buffer with a specified number of bytes.
	 * @param buffer Pointer to the byte buffer.
	 * @param nBytes Number of bytes to read from the buffer.
	 * @param isBigEndian True if the buffer is in big-endian format.
	 */
	UnsignedFixedWidthBigInt(const uint8_t* buffer, int nBytes, bool isBigEndian)
		: LongKey<N * 64>(buffer, nBytes, isBigEndian) {}

	/**
	 * @brief Constructs from a LongKey of a different size.
	 * @tparam N2 Bit size of the input LongKey.
	 * @param value The LongKey value.
	 * @param isBigEndian True if conversion should handle big-endian.
	 */
	template<int N2>
	UnsignedFixedWidthBigInt(LongKey<N2> value, bool isBigEndian = false) {
		int nBytes = N2 / 8;
		if (nBytes / 8 > N)
			nBytes = N * 8;

		if (isBigEndian) {
			for (int i = 0; i < nBytes; ++i)
				this->data.rawBytes[i] = value.data.rawBytes[nBytes - i - 1];
		} else {
			memcpy(this->data.chunks, value.data.chunks, nBytes);
		}

		bzero(&this->data.rawBytes[nBytes], N*8 - nBytes);
	}

	/**
	 * @brief Constructs from a null-terminated string (hex or decimal).
	 * @param s The string to parse.
	 */
	UnsignedFixedWidthBigInt(const char* s) : LongKey<N * 64>(s) {} // NOLINT(*-explicit-constructor)

	/**
	 * @brief Constructs from a string view (hex or decimal).
	 * @param s The string view to parse.
	 */
	explicit UnsignedFixedWidthBigInt(std::string_view s) : LongKey<N * 64>(s) {}

	/**
	 * @brief Constructs from a 64-bit unsigned integer.
	 * @param value The initial value.
	 */
	UnsignedFixedWidthBigInt(uint64_t value) { // NOLINT(*-explicit-constructor)
		this->data.chunks[0] = value;
		bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
	}

	/**
	 * @brief Constructs from a 32-bit unsigned integer.
	 * @param value The initial value.
	 */
	UnsignedFixedWidthBigInt(uint32_t value) { // NOLINT(*-explicit-constructor)
		this->data.chunks[0] = value;
		bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
	}

	/**
	 * @brief Constructs from a 16-bit unsigned integer.
	 * @param value The initial value.
	 */
	UnsignedFixedWidthBigInt(uint16_t value) { // NOLINT(*-explicit-constructor)
		this->data.chunks[0] = value;
		bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
	}

	/**
	 * @brief Constructs from a signed integer.
	 * @param value The initial value. Handles sign extension.
	 */
	UnsignedFixedWidthBigInt(int value) { // NOLINT(*-explicit-constructor)
		if (value < 0) {
			this->data.chunks[0] = (uint64_t)(int64_t)value;
			memset(&this->data.chunks[1], 0xFF, sizeof(*this->data.chunks) * (N - 1));
		} else {
			this->data.chunks[0] = value;
			bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
		}
	}

	/**
	 * @brief Constructs from a double.
	 * @param value The initial double value.
	 */
	UnsignedFixedWidthBigInt(double value) { // NOLINT(*-explicit-constructor)
		_internalBigintConvertFromDouble(value, N, this->data.chunks);
	}

	/**
	 * @brief Less-than comparison.
	 * @param other The other big integer.
	 * @return True if this < other.
	 */
	bool operator<(const UnsignedFixedWidthBigInt<N>& other) const {
		for (int i = N - 1; i >= 0; --i) {
			if (this->data.chunks[i] < other.data.chunks[i]) return true;
			if (this->data.chunks[i] > other.data.chunks[i]) return false;
		}
		return false;
	}

	/**
	 * @brief Less-than comparison with uint64_t.
	 * @param other The value to compare against.
	 * @return True if this < other.
	 */
	bool operator<(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] < other;
	}

	/**
	 * @brief Less-than comparison with uint32_t.
	 * @param other The value to compare against.
	 * @return True if this < other.
	 */
	bool operator<(uint32_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] < other;
	}

	/**
	 * @brief Less-than comparison with int.
	 * @param other The value to compare against.
	 * @return True if this < other.
	 */
	bool operator<(int other) const {
		if (other <= 0)
			return false;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] < (uint64_t)other;
	}

	/**
	 * @brief Signed less-than comparison.
	 * @param b The other big integer.
	 * @return True if this < b, treating both as signed.
	 */
	bool signedLessThan(const UnsignedFixedWidthBigInt<N>& b) const {
		bool thisNegative = this->getBit(N * 64 - 1);
		bool otherNegative = b.getBit(N * 64 - 1);

		if (thisNegative != otherNegative)
			return thisNegative;

		return *this < b;
	}

	/**
	 * @brief Greater-than comparison.
	 * @param other The other big integer.
	 * @return True if this > other.
	 */
	bool operator>(const UnsignedFixedWidthBigInt<N>& other) const {
		return other < *this;
	}

	/**
	 * @brief Greater-than comparison with uint64_t.
	 * @param other The value to compare against.
	 * @return True if this > other.
	 */
	bool operator>(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] > other;
	}

	/**
	 * @brief Greater-than comparison with int.
	 * @param other The value to compare against.
	 * @return True if this > other.
	 */
	bool operator>(int other) const {
		if (other < 0)
			return true;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] > (uint64_t)other;
	}

	/**
	 * @brief Less-than-or-equal comparison.
	 * @param other The other big integer.
	 * @return True if this <= other.
	 */
	bool operator<=(const UnsignedFixedWidthBigInt<N>& other) const {
		for (int i = N - 1; i >= 0; --i) {
			if (this->data.chunks[i] < other.data.chunks[i]) return true;
			if (this->data.chunks[i] > other.data.chunks[i]) return false;
		}
		return true;
	}

	/**
	 * @brief Less-than-or-equal comparison with uint64_t.
	 * @param other The value to compare against.
	 * @return True if this <= other.
	 */
	bool operator<=(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] <= other;
	}

	/**
	 * @brief Less-than-or-equal comparison with int.
	 * @param other The value to compare against.
	 * @return True if this <= other.
	 */
	bool operator<=(int other) const {
		if (other < 0)
			return false;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] <= (uint64_t)other;
	}

	/**
	 * @brief Greater-than-or-equal comparison.
	 * @param other The other big integer.
	 * @return True if this >= other.
	 */
	bool operator>=(const UnsignedFixedWidthBigInt<N>& other) const {
		return other <= *this;
	}

	/**
	 * @brief Greater-than-or-equal comparison with uint64_t.
	 * @param other The value to compare against.
	 * @return True if this >= other.
	 */
	bool operator>=(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] >= other;
	}

	/**
	 * @brief Greater-than-or-equal comparison with int.
	 * @param other The value to compare against.
	 * @return True if this >= other.
	 */
	bool operator>=(int other) const {
		if (other < 0)
			return true;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] >= (uint64_t)other;
	}

	/**
	 * @brief Addition operator.
	 * @param other The other big integer to add.
	 * @return The sum.
	 */
	UnsignedFixedWidthBigInt<N> operator+(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> result(*this);
		//_internalBigintFastAdd(N, this->data.chunks, N, other.data.chunks, N, result.data.chunks);
		result += other;
		return result;
	}

	/**
	 * @brief Addition operator with int.
	 * @param other The integer to add.
	 * @return The sum.
	 */
	UnsignedFixedWidthBigInt<N> operator+(int other) const {
		UnsignedFixedWidthBigInt<N> result;
		uint64_t word = (uint64_t)(int64_t)other;
		_internalBigintFastAdd(1, &word, N, this->data.chunks, N, result.data.chunks);
		return result;
	}

	/**
	 * @brief Addition assignment operator.
	 * @param other The other big integer to add.
	 * @return Reference to this.
	 */
	UnsignedFixedWidthBigInt<N>& operator+=(const UnsignedFixedWidthBigInt<N>& other) {
		uint64_t carry = 0;
		for (int i = 0; i < N; ++i) {
			uint64_t sum = this->data.chunks[i] + other.data.chunks[i] + carry;
			if (sum < this->data.chunks[i] || (carry && sum == this->data.chunks[i]))
				carry = 1;
			else
				carry = 0;
			this->data.chunks[i] = sum;
		}
		return *this;
	}

	/**
	 * @brief Addition assignment operator with int.
	 * @param other The integer to add.
	 * @return Reference to this.
	 */
	UnsignedFixedWidthBigInt<N>& operator+=(int other) {
		uint64_t word = (uint64_t)(int64_t)other;
		_internalBigintFastAdd(1, &word, N, this->data.chunks, N, this->data.chunks);
		return *this;
	}

	/**
	 * @brief Addition assignment operator with uint64_t.
	 * @param other The unsigned 64-bit integer to add.
	 * @return Reference to this.
	 */
	UnsignedFixedWidthBigInt<N>& operator+=(uint64_t other) {
		_internalBigintFastAdd(1, &other, N, this->data.chunks, N, this->data.chunks);
		return *this;
	}

	/**
	 * @brief Subtraction operator.
	 * @param other The other big integer to subtract.
	 * @return The difference.
	 */
	UnsignedFixedWidthBigInt<N> operator-(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> result(*this);
		_internalBigintFastSub(N, this->data.chunks, N, other.data.chunks, N, result.data.chunks);
		return result;
	}

	/**
	 * @brief Subtraction assignment operator.
	 * @param other The other big integer to subtract.
	 * @return Reference to this.
	 */
	UnsignedFixedWidthBigInt<N>& operator-=(const UnsignedFixedWidthBigInt<N>& other) {
		_internalBigintFastSub(N, this->data.chunks, N, other.data.chunks, N, this->data.chunks);
		return *this;
	}

	/**
	 * @brief Multiplication operator.
	 * 
	 * @details Performs unsigned multiplication of two big integers.
	 *          Uses a nested loop $O(N^2)$ approach for small numbers (N <= 32).
	 *          For larger numbers, it switches to the Karatsuba algorithm $O(N^{1.585})$.
	 * 
	 * @param other The other big integer to multiply.
	 * @return The product.
	 * 
	 * @complexity $O(N^2)$ for $N \le 32$, $O(N^{1.585})$ for $N > 32$, where $N$ is the number of 64-bit chunks.
	 */
	UnsignedFixedWidthBigInt<N> operator*(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> result;

		uint64_t* temp = result.data.chunks;

		for (int i = 0; i < N; ++i) {
			__uint128_t a = this->data.chunks[i];
			uint64_t carry = 0;

			for (int j = 0; j < N - i; ++j) {
				__uint128_t prod = a * other.data.chunks[j] + temp[i + j] + carry;
				temp[i + j] = (uint64_t)prod;
				carry = (uint64_t)(prod >> 64);
			}
		}

		return result;
	}

	/**
	 * @brief Multiplication operator with int.
	 * @param other The integer to multiply by.
	 * @return The product.
	 */
	UnsignedFixedWidthBigInt<N> operator*(int other) const {
		UnsignedFixedWidthBigInt<N> result;

		uint64_t* temp = result.data.chunks;

		uint64_t carry = 0;
		for (int i = 0; i < N; ++i) {
			__uint128_t a = this->data.chunks[i];
			__uint128_t prod = a * other + temp[i] + carry;
			temp[i] = (uint64_t)prod;
			carry = (uint64_t)(prod >> 64);
		}

		return result;
	}

	/**
	 * @brief Multiplication operator with double.
	 * @param other The double to multiply by.
	 * @return The product.
	 */
	UnsignedFixedWidthBigInt<N> operator*(double other) const {
		UnsignedFixedWidthBigInt<N> result;
		_internalBigintFastMulDouble(this->data.chunks, other, result.data.chunks, N);
		return result;
	}

	/**
	 * @brief Division operator.
	 * 
	 * @details Performs unsigned division of two big integers using a bit-by-bit long division algorithm.
	 * 
	 * @param divisor The divisor.
	 * @return The quotient.
	 * 
	 * @complexity $O(N \cdot B)$ where $N$ is the number of 64-bit chunks and $B$ is the number of bits.
	 */
	UnsignedFixedWidthBigInt<N> operator/(const UnsignedFixedWidthBigInt<N>& divisor) const {
		UnsignedFixedWidthBigInt<N> remainder(*this);
		UnsignedFixedWidthBigInt<N> quotient(0);
		_internalBigintFastDiv(remainder.data.chunks, divisor.data.chunks, quotient.data.chunks, N);
		return quotient;
	}

	/**
	 * @brief Division assignment operator.
	 * @param divisor The divisor.
	 * @return Reference to this (updated with quotient).
	 */
	UnsignedFixedWidthBigInt<N>& operator/=(const UnsignedFixedWidthBigInt<N>& divisor) const {
		UnsignedFixedWidthBigInt<N> remainder(*this);
		_internalBigintFastDiv(remainder.data.chunks, divisor.data.chunks, this->data.chunks, N);
		return *this;
	}

	/**
	 * @brief Signed division operator.
	 * 
	 * @details Performs signed division by taking the absolute values of the operands, 
	 *          performing unsigned division, and then adjusting the sign of the result.
	 *          Sign is determined by the most significant bit (two's complement).
	 * 
	 * @param denominator The divisor.
	 * @return The quotient.
	 * 
	 * @complexity $O(N \cdot B)$ due to underlying unsigned division.
	 */
	UnsignedFixedWidthBigInt<N> sdiv(const UnsignedFixedWidthBigInt<N>& denominator) const {
		if (denominator.isZero())
			return 0;

		bool thisNegative = this->getBit(N * 64 - 1);
		bool denomNegative = denominator.getBit(N * 64 - 1);

		UnsignedFixedWidthBigInt<N> absThis = thisNegative ? -(*this) : *this;
		UnsignedFixedWidthBigInt<N> absDenom = denomNegative ? -denominator : denominator;

		UnsignedFixedWidthBigInt<N> absQuotient = absThis / absDenom;

		if (thisNegative != denomNegative)
			return -absQuotient;

		return absQuotient;
	}

	/**
	 * @brief Modulo operator.
	 * @param divisor The divisor.
	 * @return The remainder.
	 */
	UnsignedFixedWidthBigInt<N> operator%(const UnsignedFixedWidthBigInt<N>& divisor) const {
		UnsignedFixedWidthBigInt<N> remainder(*this);
		_internalBigintFastDiv(remainder.data.chunks, divisor.data.chunks, nullptr, N);
		return remainder;
	}

	/**
	 * @brief Modulo assignment operator.
	 * @param divisor The divisor.
	 * @return Reference to this (updated with remainder).
	 */
	UnsignedFixedWidthBigInt<N>& operator%=(const UnsignedFixedWidthBigInt<N>& divisor) const {
		_internalBigintFastDiv(this->data.chunks, divisor.data.chunks, nullptr, N);
		return *this;
	}

	/**
	 * @brief Signed modulo operator.
	 * 
	 * @details Performs signed modulo. The sign of the result follows the sign of the dividend.
	 *          Uses unsigned modulo on absolute values and then adjusts the sign.
	 * 
	 * @param denominator The divisor.
	 * @return The remainder.
	 * 
	 * @complexity $O(N \cdot B)$ due to underlying unsigned modulo.
	 */
	UnsignedFixedWidthBigInt<N> smod(const UnsignedFixedWidthBigInt<N>& denominator) const {
		if (denominator.isZero())
			return 0;

		bool thisNegative = this->getBit(N * 64 - 1);
		UnsignedFixedWidthBigInt<N> absThis = thisNegative ? -(*this) : *this;
		UnsignedFixedWidthBigInt<N> absDenom = denominator.getBit(N * 64 - 1) ? -denominator : denominator;

		UnsignedFixedWidthBigInt<N> remainder = absThis % absDenom;

		if (thisNegative && !remainder.isZero())
			return -remainder;

		return remainder;
	}

	/**
	 * @brief Unary negation operator (two's complement).
	 * @return The negated big integer.
	 */
	UnsignedFixedWidthBigInt<N> operator-() const {
		UnsignedFixedWidthBigInt<N> result;
		uint64_t carry = 1;
		for (int i = 0; i < N; ++i) {
			result.data.chunks[i] = ~this->data.chunks[i] + carry;
			if (carry && result.data.chunks[i] == 0)
				carry = 1;
			else
				carry = 0;
		}
		return result;
	}

	/**
	 * @brief Exponentiation operator.
	 * 
	 * @details Calculates the power using the square-and-multiply algorithm.
	 * 
	 * @param power The exponent.
	 * @return This raised to the power of 'power'.
	 * 
	 * @complexity $O(\log(\text{power}) \cdot \text{MulComplexity}(N))$.
	 */
	UnsignedFixedWidthBigInt<N> pow(int power) const {
		UnsignedFixedWidthBigInt<N> out;
		_internalBigintFastPow(this->data.chunks, out.data.chunks, N, power);
		return out;
	}

	/**
	 * @brief Nth root operator.
	 * 
	 * @details For roots < 7, it uses Newton's method (fixed 8 iterations).
	 *          For roots >= 7, it uses a binary search algorithm.
	 * 
	 * @param root The root to calculate.
	 * @return The integer nth root of this.
	 * 
	 * @complexity $O(\frac{B}{root} \cdot \text{PowComplexity}(N))$ for large roots, 
	 *             or $O(\text{Iterations} \cdot (\text{PowComplexity} + \text{DivComplexity}))$ for small roots.
	 */
	UnsignedFixedWidthBigInt<N> root(int root) const {
		UnsignedFixedWidthBigInt<N> out;
		/*if (root == 2)
			_internalBigintFastSqrt(this->data.chunks, out.data.chunks, N);
		else */if (root < 7)
			_internalBigintFastSmallRoot(this->data.chunks, out.data.chunks, N, root);
		else
			_internalBigintFastRoot(this->data.chunks, out.data.chunks, N, root);
		return out;
	}

	/**
	 * @brief Sets a specific bit to 1.
	 * @param index The bit index.
	 */
	void setBit(int index) {
		int word = index / 64;
		int bit = index % 64;
		this->data.chunks[word] |= (1ULL << bit);
	}

	/**
	 * @brief Resets a specific bit to 0.
	 * @param index The bit index.
	 */
	void resetBit(int index) {
		int word = index / 64;
		int bit = index % 64;
		this->data.chunks[word] &= ~(1ULL << bit);
	}

	/**
	 * @brief Gets the value of a specific bit.
	 * @param index The bit index.
	 * @return True if the bit is set.
	 */
	bool getBit(int index) const {
		int word = index / 64;
		int bit = index % 64;
		return (this->data.chunks[word] >> bit) & 1;
	}

	/**
	 * @brief Counts the number of significant bits.
	 * @return The number of bits.
	 */
	uint16_t countBits() const {
		for (uint16_t i = N; i > 0;)
			if (this->data.chunks[--i])
				return i*64 + ::countBits(this->data.chunks[i]);
		return 0;
	}

	/**
	 * @brief Left shift assignment operator.
	 * @param n Number of bits to shift.
	 * @return Reference to this.
	 */
	UnsignedFixedWidthBigInt<N>& operator<<=(int n) {
		if (n == 0) return *this;
		if (n >= N * 64) {
			memset(this->data.chunks, 0, sizeof(this->data.chunks));
			return *this;
		}

		int wordShift = n / 64;
		int bitShift = n % 64;

		uint64_t newParts[N] = {0};

		for (int i = N - 1; i >= wordShift; --i) {
			newParts[i] |= this->data.chunks[i - wordShift] << bitShift;
			if (bitShift && i - wordShift - 1 >= 0)
				newParts[i] |= this->data.chunks[i - wordShift - 1] >> (64 - bitShift);
		}
		memcpy(this->data.chunks, newParts, sizeof(this->data.chunks));
		return *this;
	}

	/**
	 * @brief Left shift operator.
	 * @param n Number of bits to shift.
	 * @return Shifted big integer.
	 */
	UnsignedFixedWidthBigInt<N> operator<<(int n) const {
		UnsignedFixedWidthBigInt<N> out(*this);
		out <<= n;
		return out;
	}

	/**
	 * @brief Right shift assignment operator.
	 * @param n Number of bits to shift.
	 * @return Reference to this.
	 */
	UnsignedFixedWidthBigInt<N>& operator>>=(int n) {
		if (n == 0) return *this;
		if (n >= N * 64) {
			memset(this->data.chunks, 0, sizeof(this->data.chunks));
			return *this;
		}

		int wordShift = n / 64;
		int bitShift = n % 64;

		for (int i = wordShift; i < N; i++) {
			this->data.chunks[i - wordShift] = this->data.chunks[i] >> bitShift;
			if (bitShift && i + 1 < N)
				this->data.chunks[i - wordShift] |= this->data.chunks[i + 1] << (64 - bitShift);
		}

		for (int i = N - wordShift; i < N; i++)
			this->data.chunks[i] = 0;

		return *this;
	}

	/**
	 * @brief Right shift operator.
	 * @param n Number of bits to shift.
	 * @return Shifted big integer.
	 */
	UnsignedFixedWidthBigInt<N> operator>>(int n) const {
		UnsignedFixedWidthBigInt<N> out(*this);
		out >>= n;
		return out;
	}

	/**
	 * @brief Arithmetic right shift operator.
	 * @param n Number of bits to shift.
	 * @return Shifted big integer with sign extension.
	 */
	UnsignedFixedWidthBigInt<N> sar(int n) const {
		if (n == 0) return *this;

		if (getBit(N*64 - 1) == 0)
			return *this >> n;

		if (n + 1 >= N * 64) return UnsignedFixedWidthBigInt<N>(-1);

		UnsignedFixedWidthBigInt<N> out(-1);

		if ((n & 7) == 0) {
			memcpy(out.data.rawBytes, &this->data.rawBytes[n / 8], N*8 - (n/8));
			return out;
		}

		int wordShift = n / 64;
		int bitShift = n & 63;

		for (int i = wordShift; i < N; i++) {
			out.data.chunks[i - wordShift] = this->data.chunks[i] >> bitShift;
			if (i + 1 < N)
				out.data.chunks[i - wordShift] |= this->data.chunks[i + 1] << (64 - bitShift);
			else
				out.data.chunks[i - wordShift] |= (0xFFFFFFFFFFFFFFFFLU << (64 - bitShift));
		}

		return out;
	}

	/**
	 * @brief Bitwise AND operator.
	 * @param other The other big integer.
	 * @return Bitwise AND result.
	 */
	UnsignedFixedWidthBigInt<N> operator&(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = this->data.chunks[i] & other.data.chunks[i];

		return out;
	}

	/**
	 * @brief Bitwise OR operator.
	 * @param other The other big integer.
	 * @return Bitwise OR result.
	 */
	UnsignedFixedWidthBigInt<N> operator|(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = this->data.chunks[i] | other.data.chunks[i];

		return out;
	}

	/**
	 * @brief Bitwise XOR operator.
	 * @param other The other big integer.
	 * @return Bitwise XOR result.
	 */
	UnsignedFixedWidthBigInt<N> operator^(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = this->data.chunks[i] ^ other.data.chunks[i];

		return out;
	}

	/**
	 * @brief Bitwise NOT operator.
	 * @return Bitwise NOT result.
	 */
	UnsignedFixedWidthBigInt<N> operator~() const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = ~this->data.chunks[i];

		return out;
	}

	/**
	 * @brief Constrains the value between a minimum and maximum.
	 * @tparam I1 Type of minimum and return value.
	 * @tparam I2 Type of maximum.
	 * @param minimum Minimum value.
	 * @param maximum Maximum value.
	 * @return Constrained value as type I1.
	 */
	template<class I1, class I2>
	I1 constrain(I1 minimum, I2 maximum) const {
		if (UnsignedFixedWidthBigInt<N>(maximum) < *this)
			return maximum;
		else if (*this < UnsignedFixedWidthBigInt<N>(minimum))
			return minimum;
		else
			return (I1)this->data.chunks[0];
	}

	// Conversions
	/**
	 * @brief Converts to a LongKey of a different size.
	 * @tparam N2 Bit size of the result LongKey.
	 * @return The LongKey representation.
	 */
	template<int N2>
	operator LongKey<N2>() { // NOLINT(*-explicit-constructor)
		return LongKey<N2>(this->data.rawBytes, N*8);
	}

	/**
	 * @brief Converts to int.
	 * @return Lower 32/64 bits of the value as int.
	 */
	operator int() const {
		return this->data.chunks[0];
	}

	/**
	 * @brief Converts to uint32_t.
	 * @return Lower 32 bits of the value.
	 */
	operator uint32_t() const {
		return this->data.chunks[0];
	}

	/**
	 * @brief Converts to uint16_t.
	 * @return Lower 16 bits of the value.
	 */
	operator uint16_t() const {
		return this->data.chunks[0];
	}

	/**
	 * @brief Converts to uint64_t.
	 * @return Lower 64 bits of the value.
	 */
	operator uint64_t() const {
		return this->data.chunks[0];
	}

	/**
	 * @brief Converts the big integer to a double.
	 * @return The double representation.
	 */
	double toDouble() const {
		return _internalBigintConvertToDouble(countBits(), N, this->data.chunks);
	}
};

/**
 * @typedef uint128_t
 * @brief 128-bit unsigned fixed-width big integer.
 */
typedef UnsignedFixedWidthBigInt<2> uint128_t;

/**
 * @typedef uint192_t
 * @brief 192-bit unsigned fixed-width big integer.
 */
typedef UnsignedFixedWidthBigInt<3> uint192_t;

/**
 * @typedef uint256_t
 * @brief 256-bit unsigned fixed-width big integer.
 */
typedef UnsignedFixedWidthBigInt<4> uint256_t;

/**
 * @typedef uint384_t
 * @brief 384-bit unsigned fixed-width big integer.
 */
typedef UnsignedFixedWidthBigInt<6> uint384_t;

/**
 * @typedef uint512_t
 * @brief 512-bit unsigned fixed-width big integer.
 */
typedef UnsignedFixedWidthBigInt<8> uint512_t;

static_assert(!std::is_polymorphic<uint256_t>::value, "uint256_t cannot be polymorphic");


#endif //EXCESSIVE_BIGINT_H
