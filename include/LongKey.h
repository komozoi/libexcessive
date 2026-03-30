/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-07-09
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


#ifndef EXCESSIVE_LONGKEY_H
#define EXCESSIVE_LONGKEY_H

#include <cstdint>
#include <immintrin.h>
#include "string.h"
#include <string>

#include "strutil.h"


/**
 * @brief A template class for representing fixed-length keys or identifiers.
 * 
 * Provides storage, comparison, and serialization for keys of N bits.
 * 
 * @tparam N The number of bits in the key.
 */
template<int N>
class LongKey {
public:
	/**
	 * @brief Default constructor. Does not initialize the data.
	 */
	inline LongKey() {
		// Uninitialized on purpose
	}

	/**
	 * @brief Copy constructor.
	 * @param other The other LongKey to copy from.
	 */
	inline LongKey(const LongKey<N>& other) {
		memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
	}

	/**
	 * @brief Move constructor.
	 * @param other The other LongKey to move from.
	 */
	inline LongKey(LongKey<N>&& other)  noexcept {
		memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
	}

	/**
	 * @brief Constructs from a LongKey of a different size.
	 * @tparam N2 Bit size of the input LongKey.
	 * @param other The LongKey to convert from.
	 */
	template<int N2>
	inline explicit LongKey(LongKey<N2>&& other) {
		int inputSize = N2 / 8;
		if (inputSize > N / 8)
			inputSize = N / 8;
		memcpy(data.rawBytes, other.data.rawBytes, inputSize);
		bzero(&data.rawBytes[inputSize], N / 8 - inputSize);
	}

	/**
	 * @brief Constructs from a byte buffer.
	 * @param buffer Pointer to the byte buffer.
	 * @param isBigEndian True if the buffer is in big-endian format.
	 */
	LongKey(const uint8_t* buffer, bool isBigEndian = false) {
		if (isBigEndian) {
			for (int i = 0; i < N/8; ++i)
				this->data.rawBytes[i] = buffer[N/8 - i - 1];
		} else {
			memcpy(this->data.chunks, buffer, sizeof(this->data.chunks));
		}
	}

	/**
	 * @brief Constructs from a byte buffer with a specified number of bytes.
	 * @param buffer Pointer to the byte buffer.
	 * @param nBytes Number of bytes to read from the buffer.
	 * @param isBigEndian True if the buffer is in big-endian format.
	 */
	LongKey(const uint8_t* buffer, int nBytes, bool isBigEndian = false) {
		if (nBytes * 8 > N)
			nBytes = N / 8;
		if (isBigEndian) {
			for (int i = 0; i < nBytes; ++i)
				this->data.rawBytes[i] = buffer[nBytes - i - 1];
		} else {
			memcpy(this->data.chunks, buffer, nBytes);
		}

		bzero(&this->data.rawBytes[nBytes], N/8 - nBytes);
	}

	/**
	 * @brief Constructs from a hexadecimal string.
	 * @param s The null-terminated hex string (optionally starting with "0x").
	 */
	LongKey(const char* s) {
		if (s[0] == '0' && s[1] == 'x')
			s = &s[2];

		int slen = strlen(s);
		int nBytes = slen / 2;
		int zeros = N/8 - nBytes;
		if (zeros <= 0)
			nBytes = N/8;
		else
			bzero(&data.rawBytes[nBytes], zeros);

		if ((slen & 1) && nBytes < N/8)
			data.rawBytes[nBytes] = parseHexDigit(*s++);

		for (int i = 0; i < nBytes; i++) {
			uint8_t val = parseHexDigit(s[i*2]) << 4;
			val |= parseHexDigit(s[i*2 + 1]);
			data.rawBytes[nBytes - i - 1] = val;
		}

		bzero(&data.rawBytes[N/8], ((N + 63) / 64) * 8 - N/8);
	}

	/**
	 * @brief Constructs from a hexadecimal string view.
	 * @param s The hex string view (optionally starting with "0x").
	 */
	explicit LongKey(std::string_view s) {
		if (s[0] == '0' && s[1] == 'x')
			s = s.substr(2);

		int slen = s.size();
		int nBytes = slen / 2;
		int zeros = N/8 - nBytes;
		if (zeros <= 0)
			nBytes = N/8;
		else
			bzero(&data.rawBytes[nBytes], zeros);

		if ((slen & 1) && nBytes < N/8) {
			data.rawBytes[nBytes] = parseHexDigit(s[0]);
			s = s.substr(1);
		}

		for (int i = 0; i < nBytes; i++) {
			uint8_t val = parseHexDigit(s[i*2]) << 4;
			val |= parseHexDigit(s[i*2 + 1]);
			data.rawBytes[nBytes - i - 1] = val;
		}

		bzero(&data.rawBytes[N/8], ((N + 63) / 64) * 8 - N/8);
	}

	/**
	 * @brief Equality operator.
	 * @param other The other LongKey to compare with.
	 * @return True if all chunks match.
	 */
	bool operator==(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; ++i) {
			if (this->data.chunks[i] != other.data.chunks[i])
				return false;
		}
		return true;
	}

	/**
	 * @brief Inequality operator.
	 * @param other The other LongKey to compare with.
	 * @return True if any chunks differ.
	 */
	bool operator!=(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; ++i) {
			if (this->data.chunks[i] != other.data.chunks[i])
				return true;
		}
		return false;
	}

	/**
	 * @brief Less-than operator.
	 * @param other The other LongKey to compare with.
	 * @return True if this key is numerically less than the other.
	 */
	bool operator<(const LongKey<N>& other) const {
		for (int i = (N - 1) / 64; i >= 0; --i) {
			if (this->data.chunks[i] < other.data.chunks[i]) return true;
			if (this->data.chunks[i] > other.data.chunks[i]) return false;
		}
		return false;
	}

	/**
	 * @brief Greater-than operator.
	 * @param other The other LongKey to compare with.
	 * @return True if this key is numerically greater than the other.
	 */
	bool operator>(const LongKey<N>& other) const {
		return other < *this;
	}

	/**
	 * @brief Less-than-or-equal operator.
	 * @param other The other LongKey to compare with.
	 * @return True if this key is numerically less than or equal to the other.
	 */
	bool operator<=(const LongKey<N>& other) const {
		for (int i = (N - 1) / 64; i >= 0; --i) {
			if (this->data.chunks[i] <= other.data.chunks[i]) return true;
			if (this->data.chunks[i] >= other.data.chunks[i]) return false;
		}
		return false;
	}

	/**
	 * @brief Greater-than-or-equal operator.
	 * @param other The other LongKey to compare with.
	 * @return True if this key is numerically greater than or equal to the other.
	 */
	bool operator>=(const LongKey<N>& other) const {
		return other <= *this;
	}

	/**
	 * @brief Copy assignment operator.
	 * @param other The other LongKey to copy from.
	 * @return Reference to this.
	 */
	LongKey<N>& operator=(const LongKey<N>& other) {
		if (&other != this)
			memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
		return *this;
	}

	/**
	 * @brief Move assignment operator.
	 * @param other The other LongKey to move from.
	 * @return Reference to this.
	 */
	LongKey<N>& operator=(LongKey<N>&& other)  noexcept {
		memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
		return *this;
	}

	/**
	 * @brief Subtraction-like comparison operator.
	 * @param other The other LongKey to compare with.
	 * @return 1 if this > other, -1 if this < other, 0 if equal.
	 */
	long operator-(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; i++) {
			if (data.chunks[i] > other.data.chunks[i])
				return 1;
			else if (data.chunks[i] < other.data.chunks[i])
				return -1;
		}
		return 0;
	}

	/**
	 * @brief Compares this key with another.
	 * @param other The other LongKey.
	 * @return 1 if this > other, -1 if this < other, 0 if equal.
	 */
	inline int compare(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; i++) {
			if (data.chunks[i] > other.data.chunks[i])
				return 1;
			else if (data.chunks[i] < other.data.chunks[i])
				return -1;
		}
		return 0;
	}

	/**
	 * @brief Serializes the key to a byte buffer.
	 * @param buffer Output buffer (must be at least N/8 bytes).
	 * @param isBigEndian True if output should be in big-endian format.
	 */
	void toBytes(uint8_t* buffer, bool isBigEndian) const {
		if (isBigEndian) {
			for (int i = 0; i < N / 8; ++i)
				buffer[N/8 - 1 - i] = this->data.rawBytes[i];
		} else {
			memcpy(buffer, this->data.chunks, sizeof(this->data.chunks));
		}
	}

	/**
	 * @brief Converts the key to a hexadecimal string.
	 * @param buffer Output buffer for the string.
	 * @param forceSize If true, keeps leading zeros to match the full key size.
	 * @param addPrefix If true, prepends "0x" to the string.
	 */
	void toStr(char* buffer, bool forceSize = false, bool addPrefix = true) const {
		if (addPrefix) {
			buffer[0] = '0';
			buffer[1] = 'x';
		}

		int i = addPrefix ? 2 : 0, j = N/8 - 1;

		if (!forceSize)
			// Skip zeros
			for (; j > 0 && this->data.rawBytes[j] == 0; j--);

		for (; j >= 0;) {
			uint8_t value = this->data.rawBytes[j--];
			buffer[i++] = "0123456789AbCdeF"[value >> 4];
			buffer[i++] = "0123456789AbCdeF"[value & 15];
		}
		buffer[i] = 0;
	}

	/**
	 * @brief Conversion operator to std::string (hexadecimal representation).
	 * @return Hexadecimal string starting with "0x".
	 */
	operator std::string() const { // NOLINT(*-explicit-constructor)
		static constexpr char HEX_DIGITS[] = "0123456789AbcDeF";

		std::string result = "0x";
		bool started = false;

		for (int i = (N + 63)/64 - 1; i >= 0; --i) {
			for (int j = 60; j >= 0; j -= 4) {
				if (i * 64 + j >= N)
					continue;
				uint8_t nibble = (this->data.chunks[i] >> j) & 0xF;
				if (!started) {
					if (nibble == 0 && (i != 0 || j != 0)) continue; // Skip leading zeros
					started = true;
				}
				result += HEX_DIGITS[nibble];
			}
		}

		if (!started) result += '0'; // If value is zero

		return result;
	}

	/**
	 * @brief Checks if the key is all zeros.
	 * @return True if all chunks are zero.
	 */
	bool isZero() const {
		for (int i = 0; i < (N + 63) / 64; i++)
			if (data.chunks[i])
				return false;
		return true;
	}

	/*operator uint64_t() const {
		constexpr uint64_t SEED = 0x9e3779b97f4a7c15;
		constexpr uint64_t C1 = 0xff51afd7ed558ccd;
		constexpr uint64_t C2 = 0xc4ceb9fe1a85ec53;
		constexpr uint64_t C3 = 0x165667b19e3779f9;

		// Load up to 8 elements; pad with 0s if fewer
		uint64_t buffer[8] = {0};  // Zero-padded
		int wordCount = (N + 63) / 64;
		if (wordCount > 8) wordCount = 8;

		memcpy(buffer, data.chunks, wordCount * sizeof(uint64_t));
		__m512i x = _mm512_loadu_si512(buffer);

		// Initial seed vector
		__m512i acc = _mm512_set1_epi64(SEED);

		// Mix: x ^= x >> 33
		__m512i shifted = _mm512_srli_epi64(x, 33);
		x = _mm512_xor_si512(x, shifted);

		// x *= C1
		x = _mm512_mullo_epi64(x, _mm512_set1_epi64(C1));

		// x ^= x >> 33
		shifted = _mm512_srli_epi64(x, 33);
		x = _mm512_xor_si512(x, shifted);

		// x *= C2
		x = _mm512_mullo_epi64(x, _mm512_set1_epi64(C2));

		// x ^= x >> 33
		shifted = _mm512_srli_epi64(x, 33);
		x = _mm512_xor_si512(x, shifted);

		// acc ^= x
		acc = _mm512_xor_si512(acc, x);

		// acc = rotl(acc, 13)
		__m512i rotl = _mm512_or_si512(_mm512_slli_epi64(acc, 13), _mm512_srli_epi64(acc, 64 - 13));
		acc = rotl;

		// acc *= C3
		acc = _mm512_mullo_epi64(acc, _mm512_set1_epi64(C3));

		// Horizontal XOR reduction to uint64_t
		alignas(64) uint64_t lanes[8];
		_mm512_store_si512((__m512i*)lanes, acc);

		uint64_t hash = lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3] ^
						lanes[4] ^ lanes[5] ^ lanes[6] ^ lanes[7];

		// Final avalanche
		hash ^= hash >> 33;
		hash *= C1;
		hash ^= hash >> 33;

		return hash;
	}*/

	/**
	 * @brief 64-bit hash operator (avalanche hash).
	 * @return 64-bit hash value.
	 */
	operator uint64_t() const {
		constexpr uint64_t SEED = 0x9e3779b97f4a7c15;
		constexpr uint64_t C1 = 0xff51afd7ed558ccd;
		constexpr uint64_t C2 = 0xc4ceb9fe1a85ec53;
		constexpr uint64_t C3 = 0x165667b19e3779f9;

		// Load up to 8 elements; zero-pad if fewer
		uint64_t buffer[8] = {0};
		int wordCount = (N + 63) / 64;
		if (wordCount > 8) wordCount = 8;

		memcpy(buffer, data.chunks, wordCount * sizeof(uint64_t));

		// Accumulator initialization
		uint64_t acc[8];
		for (int i = 0; i < 8; ++i) acc[i] = SEED;

		// Mixing loop (same steps as AVX-512, per element)
		for (int i = 0; i < 8; ++i) {
			uint64_t x = buffer[i];

			x ^= x >> 33;
			x *= C1;
			x ^= x >> 33;
			x *= C2;
			x ^= x >> 33;

			acc[i] ^= x;

			// Rotate left 13
			acc[i] = (acc[i] << 13) | (acc[i] >> (64 - 13));
			acc[i] *= C3;
		}

		// Horizontal XOR reduction
		uint64_t hash = acc[0] ^ acc[1] ^ acc[2] ^ acc[3] ^
						acc[4] ^ acc[5] ^ acc[6] ^ acc[7];

		// Final avalanche
		hash ^= hash >> 33;
		hash *= C1;
		hash ^= hash >> 33;

		return hash;
	}

	/**
	 * @brief Union for accessing the key data as chunks or raw bytes.
	 */
	union {
		uint64_t chunks[(N + 63) / 64]; /**< Data as 64-bit chunks. */
		uint8_t rawBytes[((N + 63) / 64) * 8]; /**< Data as raw bytes. */
	} data{};
};

/**
 * @brief Output stream operator for LongKey.
 * @tparam N Bit size.
 * @param os Output stream.
 * @param val LongKey to output.
 * @return Reference to the output stream.
 */
template<int N>
std::ostream& operator<<(std::ostream& os, const LongKey<N>& val) {
	os << (std::string)val;
	return os;
}

/**
 * @brief Specialization of std::hash for LongKey.
 * @tparam N Bit size.
 */
template <int N>
struct std::hash<LongKey<N>> {
	/**
	 * @brief Hash function for LongKey.
	 * @param k The key to hash.
	 * @return Hash value.
	 */
	std::size_t operator()(const LongKey<N>& k) const {
		return k.operator uint64_t();
	}
};

#endif //EXCESSIVE_LONGKEY_H
