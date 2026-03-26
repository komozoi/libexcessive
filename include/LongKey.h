//
// Created by komozoi on 9.7.25.
//

#ifndef EXCESSIVE_LONGKEY_H
#define EXCESSIVE_LONGKEY_H

#include <cstdint>
#include <immintrin.h>
#include "string.h"
#include <string>

#include "strutil.h"


template<int N>
class LongKey {
public:
	inline LongKey() {
		// Uninitialized on purpose
	}

	inline LongKey(const LongKey<N>& other) {
		memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
	}

	inline LongKey(LongKey<N>&& other)  noexcept {
		memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
	}

	template<int N2>
	inline explicit LongKey(LongKey<N2>&& other) {
		int inputSize = N2 / 8;
		if (inputSize > N / 8)
			inputSize = N / 8;
		memcpy(data.rawBytes, other.data.rawBytes, inputSize);
		bzero(&data.rawBytes[inputSize], N / 8 - inputSize);
	}

	LongKey(const uint8_t* buffer, bool isBigEndian = false) {
		if (isBigEndian) {
			for (int i = 0; i < N/8; ++i)
				this->data.rawBytes[i] = buffer[N/8 - i - 1];
		} else {
			memcpy(this->data.chunks, buffer, sizeof(this->data.chunks));
		}
	}

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

	// Equality
	bool operator==(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; ++i) {
			if (this->data.chunks[i] != other.data.chunks[i])
				return false;
		}
		return true;
	}

	bool operator!=(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; ++i) {
			if (this->data.chunks[i] != other.data.chunks[i])
				return true;
		}
		return false;
	}

	// Comparison
	bool operator<(const LongKey<N>& other) const {
		for (int i = (N - 1) / 64; i >= 0; --i) {
			if (this->data.chunks[i] < other.data.chunks[i]) return true;
			if (this->data.chunks[i] > other.data.chunks[i]) return false;
		}
		return false;
	}

	bool operator>(const LongKey<N>& other) const {
		return other < *this;
	}

	bool operator<=(const LongKey<N>& other) const {
		for (int i = (N - 1) / 64; i >= 0; --i) {
			if (this->data.chunks[i] <= other.data.chunks[i]) return true;
			if (this->data.chunks[i] >= other.data.chunks[i]) return false;
		}
		return false;
	}

	bool operator>=(const LongKey<N>& other) const {
		return other <= *this;
	}

	LongKey<N>& operator=(const LongKey<N>& other) {
		if (&other != this)
			memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
		return *this;
	}

	LongKey<N>& operator=(LongKey<N>&& other)  noexcept {
		memcpy(data.rawBytes, other.data.rawBytes, sizeof(data));
		return *this;
	}

	long operator-(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; i++) {
			if (data.chunks[i] > other.data.chunks[i])
				return 1;
			else if (data.chunks[i] < other.data.chunks[i])
				return -1;
		}
		return 0;
	}

	inline int compare(const LongKey<N>& other) const {
		for (int i = 0; i < (N + 63) / 64; i++) {
			if (data.chunks[i] > other.data.chunks[i])
				return 1;
			else if (data.chunks[i] < other.data.chunks[i])
				return -1;
		}
		return 0;
	}

	// Serialize to byte buffer
	void toBytes(uint8_t* buffer, bool isBigEndian) const {
		if (isBigEndian) {
			for (int i = 0; i < N / 8; ++i)
				buffer[N/8 - 1 - i] = this->data.rawBytes[i];
		} else {
			memcpy(buffer, this->data.chunks, sizeof(this->data.chunks));
		}
	}

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

	union {
		uint64_t chunks[(N + 63) / 64];
		uint8_t rawBytes[((N + 63) / 64) * 8];
	} data{};
};

// Output operator for this
template<int N>
std::ostream& operator<<(std::ostream& os, const LongKey<N>& val) {
	os << (std::string)val;
	return os;
}

template <int N>
struct std::hash<LongKey<N>> {
	std::size_t operator()(const LongKey<N>& k) const {
		return k.operator uint64_t();
	}
};

#endif //EXCESSIVE_LONGKEY_H
