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


template<int N>
class UnsignedFixedWidthBigInt: public LongKey<N*64> {
public:
	UnsignedFixedWidthBigInt() = default;

	// Construct from byte buffer
	UnsignedFixedWidthBigInt(const uint8_t* buffer, bool isBigEndian)
		: LongKey<N * 64>(buffer, isBigEndian) {}

	UnsignedFixedWidthBigInt(const uint8_t* buffer, int nBytes, bool isBigEndian)
		: LongKey<N * 64>(buffer, nBytes, isBigEndian) {}

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

	UnsignedFixedWidthBigInt(const char* s) : LongKey<N * 64>(s) {} // NOLINT(*-explicit-constructor)

	explicit UnsignedFixedWidthBigInt(std::string_view s) : LongKey<N * 64>(s) {}

	UnsignedFixedWidthBigInt(uint64_t value) { // NOLINT(*-explicit-constructor)
		this->data.chunks[0] = value;
		bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
	}

	UnsignedFixedWidthBigInt(uint32_t value) { // NOLINT(*-explicit-constructor)
		this->data.chunks[0] = value;
		bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
	}

	UnsignedFixedWidthBigInt(uint16_t value) { // NOLINT(*-explicit-constructor)
		this->data.chunks[0] = value;
		bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
	}

	UnsignedFixedWidthBigInt(int value) { // NOLINT(*-explicit-constructor)
		if (value < 0) {
			this->data.chunks[0] = (uint64_t)(int64_t)value;
			memset(&this->data.chunks[1], 0xFF, sizeof(*this->data.chunks) * (N - 1));
		} else {
			this->data.chunks[0] = value;
			bzero(&this->data.chunks[1], sizeof(*this->data.chunks) * (N - 1));
		}
	}

	UnsignedFixedWidthBigInt(double value) { // NOLINT(*-explicit-constructor)
		_internalBigintConvertFromDouble(value, N, this->data.chunks);
	}

	// Comparison
	bool operator<(const UnsignedFixedWidthBigInt<N>& other) const {
		for (int i = N - 1; i >= 0; --i) {
			if (this->data.chunks[i] < other.data.chunks[i]) return true;
			if (this->data.chunks[i] > other.data.chunks[i]) return false;
		}
		return false;
	}

	bool operator<(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] < other;
	}

	bool operator<(uint32_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] < other;
	}

	bool operator<(int other) const {
		if (other <= 0)
			return false;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] < (uint64_t)other;
	}

	bool signedLessThan(const UnsignedFixedWidthBigInt<N>& b) const {
		bool thisNegative = this->getBit(N * 64 - 1);
		bool otherNegative = b.getBit(N * 64 - 1);

		if (thisNegative != otherNegative)
			return thisNegative;

		return *this < b;
	}

	bool operator>(const UnsignedFixedWidthBigInt<N>& other) const {
		return other < *this;
	}

	bool operator>(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] > other;
	}

	bool operator>(int other) const {
		if (other < 0)
			return true;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] > (uint64_t)other;
	}

	bool operator<=(const UnsignedFixedWidthBigInt<N>& other) const {
		for (int i = N - 1; i >= 0; --i) {
			if (this->data.chunks[i] < other.data.chunks[i]) return true;
			if (this->data.chunks[i] > other.data.chunks[i]) return false;
		}
		return true;
	}

	bool operator<=(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] <= other;
	}

	bool operator<=(int other) const {
		if (other < 0)
			return false;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return false;
		return this->data.chunks[0] <= (uint64_t)other;
	}

	bool operator>=(const UnsignedFixedWidthBigInt<N>& other) const {
		return other <= *this;
	}

	bool operator>=(uint64_t other) const {
		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] >= other;
	}

	bool operator>=(int other) const {
		if (other < 0)
			return true;

		for (int i = N - 1; i > 0; --i)
			if (this->data.chunks[i])
				return true;
		return this->data.chunks[0] >= (uint64_t)other;
	}

	// Addition
	UnsignedFixedWidthBigInt<N> operator+(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> result(*this);
		//_internalBigintFastAdd(N, this->data.chunks, N, other.data.chunks, N, result.data.chunks);
		result += other;
		return result;
	}

	UnsignedFixedWidthBigInt<N> operator+(int other) const {
		UnsignedFixedWidthBigInt<N> result;
		uint64_t word = (uint64_t)(int64_t)other;
		_internalBigintFastAdd(1, &word, N, this->data.chunks, N, result.data.chunks);
		return result;
	}

	// Addition
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

	// Addition
	UnsignedFixedWidthBigInt<N>& operator+=(int other) {
		uint64_t word = (uint64_t)(int64_t)other;
		_internalBigintFastAdd(1, &word, N, this->data.chunks, N, this->data.chunks);
		return *this;
	}

	// Addition
	UnsignedFixedWidthBigInt<N>& operator+=(uint64_t other) {
		_internalBigintFastAdd(1, &other, N, this->data.chunks, N, this->data.chunks);
		return *this;
	}

	// Subtraction
	UnsignedFixedWidthBigInt<N> operator-(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> result(*this);
		_internalBigintFastSub(N, this->data.chunks, N, other.data.chunks, N, result.data.chunks);
		return result;
	}

	// Subtraction
	UnsignedFixedWidthBigInt<N>& operator-=(const UnsignedFixedWidthBigInt<N>& other) {
		_internalBigintFastSub(N, this->data.chunks, N, other.data.chunks, N, this->data.chunks);
		return *this;
	}

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

	UnsignedFixedWidthBigInt<N> operator*(double other) const {
		UnsignedFixedWidthBigInt<N> result;
		_internalBigintFastMulDouble(this->data.chunks, other, result.data.chunks, N);
		return result;
	}

	UnsignedFixedWidthBigInt<N> operator/(const UnsignedFixedWidthBigInt<N>& divisor) const {
		UnsignedFixedWidthBigInt<N> remainder(*this);
		UnsignedFixedWidthBigInt<N> quotient(0);
		_internalBigintFastDiv(remainder.data.chunks, divisor.data.chunks, quotient.data.chunks, N);
		return quotient;
	}

	UnsignedFixedWidthBigInt<N>& operator/=(const UnsignedFixedWidthBigInt<N>& divisor) const {
		UnsignedFixedWidthBigInt<N> remainder(*this);
		_internalBigintFastDiv(remainder.data.chunks, divisor.data.chunks, this->data.chunks, N);
		return *this;
	}

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

	UnsignedFixedWidthBigInt<N> operator%(const UnsignedFixedWidthBigInt<N>& divisor) const {
		UnsignedFixedWidthBigInt<N> remainder(*this);
		_internalBigintFastDiv(remainder.data.chunks, divisor.data.chunks, nullptr, N);
		return remainder;
	}

	UnsignedFixedWidthBigInt<N>& operator%=(const UnsignedFixedWidthBigInt<N>& divisor) const {
		_internalBigintFastDiv(this->data.chunks, divisor.data.chunks, nullptr, N);
		return *this;
	}

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

	UnsignedFixedWidthBigInt<N> pow(int power) const {
		UnsignedFixedWidthBigInt<N> out;
		_internalBigintFastPow(this->data.chunks, out.data.chunks, N, power);
		return out;
	}

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

	void setBit(int index) {
		int word = index / 64;
		int bit = index % 64;
		this->data.chunks[word] |= (1ULL << bit);
	}

	void resetBit(int index) {
		int word = index / 64;
		int bit = index % 64;
		this->data.chunks[word] &= ~(1ULL << bit);
	}

	bool getBit(int index) const {
		int word = index / 64;
		int bit = index % 64;
		return (this->data.chunks[word] >> bit) & 1;
	}

	uint16_t countBits() const {
		for (uint16_t i = N; i > 0;)
			if (this->data.chunks[--i])
				return i*64 + ::countBits(this->data.chunks[i]);
		return 0;
	}

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

	UnsignedFixedWidthBigInt<N> operator<<(int n) const {
		UnsignedFixedWidthBigInt<N> out(*this);
		out <<= n;
		return out;
	}

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

	UnsignedFixedWidthBigInt<N> operator>>(int n) const {
		UnsignedFixedWidthBigInt<N> out(*this);
		out >>= n;
		return out;
	}

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

	UnsignedFixedWidthBigInt<N> operator&(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = this->data.chunks[i] & other.data.chunks[i];

		return out;
	}

	UnsignedFixedWidthBigInt<N> operator|(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = this->data.chunks[i] | other.data.chunks[i];

		return out;
	}

	UnsignedFixedWidthBigInt<N> operator^(const UnsignedFixedWidthBigInt<N>& other) const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = this->data.chunks[i] ^ other.data.chunks[i];

		return out;
	}

	UnsignedFixedWidthBigInt<N> operator~() const {
		UnsignedFixedWidthBigInt<N> out;

		for (int i = 0; i < N; i++)
			out.data.chunks[i] = ~this->data.chunks[i];

		return out;
	}

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
	template<int N2>
	operator LongKey<N2>() { // NOLINT(*-explicit-constructor)
		return LongKey<N2>(this->data.rawBytes, N*8);
	}

	operator int() const {
		return this->data.chunks[0];
	}

	operator uint32_t() const {
		return this->data.chunks[0];
	}

	operator uint16_t() const {
		return this->data.chunks[0];
	}

	operator uint64_t() const {
		return this->data.chunks[0];
	}

	double toDouble() const {
		return _internalBigintConvertToDouble(countBits(), N, this->data.chunks);
	}
};


typedef UnsignedFixedWidthBigInt<2> uint128_t;
typedef UnsignedFixedWidthBigInt<3> uint192_t;
typedef UnsignedFixedWidthBigInt<4> uint256_t;
typedef UnsignedFixedWidthBigInt<6> uint384_t;
typedef UnsignedFixedWidthBigInt<8> uint512_t;

static_assert(!std::is_polymorphic<uint256_t>::value, "uint256_t cannot be polymorphic");


#endif //EXCESSIVE_BIGINT_H
