//
// Created by komozoi on 29.7.25.
//

#include "bigint.h"


#define IS_UINT64_OVERFLOW(x) ((x >> 64) > 0)
#define MIN(a, b) (a > b ? (b) : (a))
#define MAX(a, b) (a < b ? (b) : (a))


// Most functions here modified from https://github.com/ilia3101/Big-Integer-C/blob/master/BigInt.c#L371
// Includes most of the original author's comments


static void _lshift_word(int size, uint64_t* a, int nwords) {
	int_fast32_t i;
	/* Shift whole words */
	for (i = (size - 1); i >= nwords; --i)
		a[i] = a[i - nwords];
	/* Zero pad shifted words. */
	for (; i >= 0; --i)
		a[i] = 0;
}


void _internalBigintFastAdd(int aSize, const uint64_t* a, int bSize, const uint64_t* b, int ySize, uint64_t* y) {
	/* Make it so that A will be smaller than B */
	if (aSize > bSize) {
		int temp1 = bSize;
		bSize = aSize;
		aSize = temp1;
		const uint64_t* temp2 = b;
		b = a;
		a = temp2;
	}

	int loop_to = 0;
	int loop1 = 0;
	int loop2 = 0;
	int loop3 = 0;

	if (ySize <= aSize) {
		loop_to = 1;
		loop1 = ySize;
	} else if (ySize <= bSize) {
		loop_to = 2;
		loop1 = aSize;
		loop2 = ySize;
	} else {
		loop_to = 3;
		loop1 = aSize;
		loop2 = bSize;
		loop3 = ySize;
	}

	int carry = 0;
	__uint128_t tmp;
	int i;

	for (i = 0; i < loop1; ++i) {
		tmp = (__uint128_t)a[i] + b[i] + carry;
		carry = IS_UINT64_OVERFLOW(tmp);
		y[i] = (tmp & UINT64_MAX);
	}

	if (loop_to == 1) return;

	for (; i < loop2; ++i) {
		tmp = (__uint128_t)b[i] + 0 + carry;
		carry = IS_UINT64_OVERFLOW(tmp);
		y[i] = (tmp & UINT64_MAX);
	}

	if (loop_to == 2) return;

	/* Do the carry, then fill the rest with zeros */
	y[i++] = carry;

	for (; i < loop3; ++i) y[i] = 0;
}


void _internalBigintFastSub(int aSize, const uint64_t* a, int bSize, const uint64_t* b, int ySize, uint64_t* y) {
	if (aSize == bSize && bSize == ySize) {
		uint64_t borrow = 0;
		for (int i = 0; i < aSize; ++i) {
			uint64_t difference = a[i] - b[i] - borrow;
			if (difference > a[i] || (borrow && difference == a[i]))
				borrow = 1;
			else
				borrow = 0;
			y[i] = difference;
		}
		return;
	}

	int loop_to = 0;
	int loop1 = 0;
	int loop2 = 0;
	int loop3 = 0;

	if (ySize <= MIN(aSize, bSize)) {
		loop_to = 1;
		loop1 = MIN(aSize, bSize);
	} else if (ySize <= MAX(aSize, bSize)) {
		loop_to = 2;
		loop1 = MIN(aSize, bSize);
		loop2 = ySize;
	} else {
		loop_to = 3;
		loop1 = aSize;
		loop2 = bSize;
		loop3 = ySize;
	}

	__uint128_t res;
	__uint128_t tmp1;
	__uint128_t tmp2;
	int borrow = 0;
	int i;

	for (i = 0; i < loop1; ++i) {
		tmp1 = (__uint128_t)a[i] + ((__uint128_t)UINT64_MAX + 1); /* + number_base */
		tmp2 = (__uint128_t)b[i] + borrow;
		res = (tmp1 - tmp2);
		y[i] = (uint64_t)(res & UINT64_MAX);
		/* "modulo number_base" == "%(number_base - 1)" if number_base is 2^N */
		borrow = !IS_UINT64_OVERFLOW(res);
	}

	if (loop_to == 1) return;

	if (aSize > bSize) {
		for (; i < loop2; ++i) {
			tmp1 = (__uint128_t)a[i] + ((__uint128_t)UINT64_MAX + 1);
			tmp2 =  borrow;
			res = (tmp1 - tmp2);
			y[i] = (uint64_t)(res & UINT64_MAX);
			borrow = !IS_UINT64_OVERFLOW(res);
		}
	} else {
		for (; i < loop2; ++i) {
			tmp1 = (__uint128_t)UINT64_MAX + 1;
			tmp2 = (__uint128_t)b[i] + borrow;
			res = (tmp1 - tmp2);
			y[i] = (uint64_t)(res & UINT64_MAX);
			borrow = !IS_UINT64_OVERFLOW(res);
		}
	}

	if (loop_to == 2) return;

	for (; i < loop3; ++i) {
		tmp1 = (__uint128_t)0 + (UINT64_MAX + 1);
		tmp2 = (__uint128_t)0 + borrow;
		res = (tmp1 - tmp2);
		y[i] = (uint64_t)(res & UINT64_MAX);
		borrow = !IS_UINT64_OVERFLOW(res);
	}
}


void _internalBigintKaratsuba(int aSize, const uint64_t* a, int bSize, const uint64_t* b, int ySize, uint64_t* y, int rlevel = 0);

void _internalBigintFastMul(int size, const uint64_t* a, const uint64_t* b, uint64_t* y) {
	uint64_t row[size];
	uint64_t tmp[size];
	int i, j;

	if (size > 32) {
		_internalBigintKaratsuba(size, a, size, b, size, y);
		return;
	}

	bzero(y, sizeof(uint64_t) * size);

	for (i = 0; i < size; ++i) {
		bzero(row, sizeof(uint64_t) * size);

		for (j = 0; j < size; ++j) {
			if (i + j < size) {
				bzero(tmp, sizeof(uint64_t) * size);
				__uint128_t intermediate = ((__uint128_t)a[i] * (__uint128_t)b[j]);
				if (i + j < size)
					tmp[i + j] = intermediate & 0xFFFFFFFFFFFFFFFFUL;
				if (i + j + 1 < size)
					tmp[i + j + 1] = intermediate >> 64;
				_internalBigintFastAdd(size, tmp, size, row, size, row);
			}
		}

		_internalBigintFastAdd(size, y, size, row, size, y);
	}
}


// Cool USSR algorithm for fast multiplication (THERE IS NOT A SINGLE 100% CORRECT PSEUDO CODE ONLINE)
void _internalBigintKaratsuba(int aSize, const uint64_t* a, int bSize, const uint64_t* b, int ySize, uint64_t* y, int rlevel) {
	/* Optimise the size, to avoid wasting any resources */
	// Must be done by caller!
	/*aSize = BigInt_truncate(aSize, num1);
	bSize = BigInt_truncate(bSize, num2);*/

	if (aSize == 0 || bSize == 0) {
		bzero(y, sizeof(uint64_t) * ySize);
		return;
	}

	if (aSize == 1 && bSize == 1) {
		__uint128_t result = ((__uint128_t)(*a)) * ((__uint128_t)(*b));
		if (ySize > 0)
			y[0] = result & 0xFFFFFFFFFFFFFFFFUL;
		if (ySize > 1)
			y[1] = result >> 64;
		if (ySize > 2)
			bzero(&y[2], sizeof(uint64_t) * (ySize - 2));
		return;
	}

	int m = MIN(aSize, bSize);
	int m2 = m / 2;
	/* do A round up, this is what stops infinite recursion when the inputs are size 1 and 2 */
	if ((m % 2) == 1) ++m2;

	/* low 1 */
	int low1Size = m2;
	const uint64_t* low1 = a;
	/* high 1 */
	int high1Size = aSize - m2;
	const uint64_t* high1 = &a[m2];
	/* low 2 */
	int low2Size = m2;
	const uint64_t* low2 = b;
	/* high 2 */
	int high2Size = bSize - m2;
	const uint64_t* high2 = &b[m2];

	// z0 = karatsuba(low1, low2)
	// z1 = karatsuba((low1 + high1), (low2 + high2))
	// z2 = karatsuba(high1, high2)
	int z0Size = low1Size + low2Size;
	uint64_t z0[z0Size];
	int z1Size = (MAX(low1Size, high1Size) + 1) + (MAX(low2Size, high2Size) + 1);
	uint64_t z1[z1Size];
	int z2Size =  high1Size + high2Size;

	/* Sometimes we can use y to store z2, then we don't have to copy from z2 to out later (2X SPEEDUP!) */
	int use_out_as_z2 = (ySize >= z2Size);
	if (use_out_as_z2)
		/* The remaining part of y must be ZERO'D */
		bzero(&y[z2Size], sizeof(uint64_t) * (ySize - z2Size));

	uint64_t* z2 = (use_out_as_z2) ? y : (uint64_t*)alloca(z2Size * sizeof(uint64_t));

	/* Make z0 and z2 */
	_internalBigintKaratsuba(low1Size, low1, low2Size, low2, z0Size, z0, rlevel+1);
	_internalBigintKaratsuba(high1Size, high1, high2Size, high2, z2Size, z2, rlevel+1);

	/* make z1 */
	{
		int low1high1Size = MAX(low1Size, high1Size)+1;
		int low2high2Size = MAX(low2Size, high2Size)+1;
		uint64_t low1high1[low1high1Size];
		uint64_t low2high2[low2high2Size];
		_internalBigintFastAdd(low1Size, low1, high1Size, high1, low1high1Size, low1high1);
		_internalBigintFastAdd(low2Size, low2, high2Size, high2, low2high2Size, low2high2);
		_internalBigintKaratsuba(low1high1Size, low1high1, low2high2Size, low2high2, z1Size, z1, rlevel+1);
	}

	// return (z2 * 10 ^ (m2 * 2)) + ((z1 - z2 - z0) * 10 ^ m2) + z0
	_internalBigintFastSub(z1Size, z1, z2Size, z2, z1Size, z1);
	_internalBigintFastSub(z1Size, z1, z0Size, z0, z1Size, z1);
	if (!use_out_as_z2) {
		int i;
		int smallest = MIN(ySize, z2Size);
		for (i = 0; i < smallest; ++i) y[i] = z2[i];
		for (; i < ySize; ++i) y[i] = 0;
	}

	_lshift_word(ySize, y, m2);
	_internalBigintFastAdd(z1Size, z1, ySize, y, ySize, y);
	_lshift_word(ySize, y, m2);
	_internalBigintFastAdd(ySize, y, z0Size, z0, ySize, y);
}


void _internalBigintFastDiv(uint64_t* remainder, const uint64_t* divisor, uint64_t* quotient, int size) {
	uint16_t numeratorBits = 0;
	for (int i = size - 1; i >= 0; i--)
		if (remainder[i]) {
			numeratorBits = countBits(remainder[i]) + i*64;
			break;
		}

	uint16_t divisorBits = 0;
	for (int i = size - 1; i >= 0; i--)
		if (divisor[i]) {
			divisorBits = countBits(divisor[i]) + i*64;
			break;
		}

	if (divisorBits == 0) {
		bzero(remainder, size*8);
		bzero(quotient, size*8);
		return;
	}// else if (numeratorBits == divisorBits && remainder[(numeratorBits - 1) / 64] < divisor[(divisorBits - 1) / 64])
	//	return;

	while (numeratorBits >= divisorBits) {
		uint16_t bits = numeratorBits - divisorBits;
		uint16_t bitShift1 = bits & 63;
		uint16_t bitShift2 = 64 - bitShift1;

		// This heuristic decreases the likelyhood of a future revert addition
		// It's a speed optimization
		if (bits == 0) {
			bool shouldExit = false;
			for (int i = (divisorBits - 1) >> 6; i >= 0; i--)
				if (remainder[i] < divisor[i]) {
					shouldExit = true;
					break;
				} else if (remainder[i] > divisor[i])
					break;

			if (shouldExit)
				break;
		} else if (remainder[(numeratorBits - 1) >> 6] < divisor[(divisorBits - 1) >> 6] >> bitShift2) {
			bits--;
			bitShift1 = bits & 63;
			bitShift2 = 64 - bitShift1;
		}

		// Combined shl/subtract with overflow detection and bit counting
		uint64_t borrow = 0;
		uint16_t wordShift = bits >> 6;
		uint16_t numeratorBitsOld = numeratorBits;
		uint16_t bitcntIdx = 0;
		for (uint16_t i = wordShift; (i*64 < numeratorBitsOld || borrow) && i < size; i++) {
			uint64_t subtractor = divisor[i - wordShift] << bitShift1;
			if (bitShift1 && i > wordShift)
				subtractor |= divisor[i - wordShift - 1] >> bitShift2;

			uint64_t temp = remainder[i] - subtractor - borrow;
			if (remainder[i] < temp)
				borrow = 1;
			else if (remainder[i] == temp)
				; //borrow = borrow;
			else
				borrow = 0;

			remainder[i] = temp;
			if (temp != 0)
				bitcntIdx = i;
		}

		if (borrow) {
			// Go back - we subtracted one more than we were supposed to
			bits--;
			uint64_t carry = 0;
			wordShift = bits / 64;
			bitShift1 = bits & 63;
			bitShift2 = 64 - bitShift1;
			for (uint16_t i = wordShift; (i*64 < numeratorBitsOld || carry) && i < size; i++) {
				uint64_t adder = divisor[i - wordShift] << bitShift1;
				if (bitShift1 && i > wordShift)
					adder |= divisor[i - wordShift - 1] >> bitShift2;

				uint64_t temp = remainder[i] + adder + carry;
				if (remainder[i] > temp)
					carry = 1;
				else if (remainder[i] == temp)
					; //carry = carry;
				else
					carry = 0;

				remainder[i] = temp;
				if (temp)
					bitcntIdx = i;
			}
		}

		numeratorBits = ::countBits(remainder[bitcntIdx]) + 64*bitcntIdx;

		if (quotient)
			quotient[wordShift] |= (uint64_t)1 << bitShift1;
	}
}

void _internalBigintFastPow(const uint64_t* x, uint64_t* y, int size, int power) {
	uint64_t multiplier[size];
	uint64_t tmp[size];
	memcpy(multiplier, x, size * sizeof(uint64_t));

	bzero(&y[1], (size - 1) * sizeof(uint64_t));
	y[0] = 1;

	while (power) {
		if (power & 1) {
			_internalBigintFastMul(size, y, multiplier, tmp);
			memcpy(y, tmp, size * sizeof(uint64_t));
		}

		if (power > 1) {
			_internalBigintFastMul(size, multiplier, multiplier, tmp);
			memcpy(multiplier, tmp, size * sizeof(uint64_t));
		} else
			break;

		power = power >> 1;
	}
}

void _internalBigintFastRoot(const uint64_t* x, uint64_t* y, int size, int root) {
	if (root == 1)
		memcpy(y, x, size * sizeof(uint64_t));
	else if (root <= 0)
		memset(y, 0xFF, size * sizeof(uint64_t));
	else {
		uint64_t* high = y;
		bool zero = true;
		uint16_t inputBits = 0;
		for (int i = size - 1; i >= 0; i--)
			if (zero && x[i] != 0) {
				zero = false;
				inputBits = countBits(x[i]) + 64 * i;
			}
		if (zero)
			return;
		if (inputBits == 1) {
			// root of 1 is always 1
			memcpy(y, x, size * sizeof(uint64_t));
			return;
		}

		uint16_t hShift = inputBits - inputBits / root - 2;
		uint16_t lShift = hShift + 3;

		uint16_t wordShift = hShift / 64;
		uint16_t bitShift = hShift % 64;

		for (int i = wordShift; i < size; i++) {
			high[i - wordShift] = x[i] >> bitShift;
			if (bitShift && i + 1 < size)
				high[i - wordShift] |= x[i + 1] << (64 - bitShift);
		}

		for (int i = size - wordShift; i < size; i++)
			high[i] = 0;

		uint64_t low[size];

		wordShift = lShift / 64;
		bitShift = lShift % 64;

		for (int i = wordShift; i < size; i++) {
			low[i - wordShift] = x[i] >> bitShift;
			if (bitShift && i + 1 < size)
				low[i - wordShift] |= x[i + 1] << (64 - bitShift);
		}

		for (int i = size - wordShift; i < size; i++)
			low[i] = 0;

		uint64_t mid[size];
		uint64_t midPow[size];

		bool keepRunning = true;
		while (keepRunning) {
			// Fast add with right shift
			uint64_t carry = 0;
			for (int i = 0; i < size; ++i) {
				uint64_t sum = low[i] + high[i] + carry;
				if (sum < low[i] || (carry && sum == low[i]))
					carry = 1;
				else
					carry = 0;

				mid[i] = sum >> 1;
				if (i)
					mid[i - 1] |= sum << 63;
			}

			_internalBigintFastPow(mid, midPow, size, root);

			// Fast comparison
			uint64_t toAdd = 0;
			uint64_t* midDst = nullptr;
			for (int i = size - 1; i >= 0; i--) {
				if (midPow[i] < x[i]) {
					midDst = low;
					toAdd = 1;
					break;
				} else if (midPow[i] > x[i]) {
					midDst = high;
					toAdd = 0xFFFFFFFFFFFFFFFFUL;
					break;
				}
			}

			// midDst will be null if midPow and x are equal
			if (midDst == nullptr) {
				// If midPow is equal to x, then we found a perfect power, return immediately
				memcpy(y, mid, size * sizeof(uint64_t));
				break;
			}

			// Perform increment/decrement-load (midDst = mid +/- 1)
			carry = 0;
			for (int i = 0; i < size; i++) {
				uint64_t sum = mid[i] + toAdd + carry;
				if (sum < mid[i] || (carry && sum == mid[i]))
					carry = 1;
				else
					carry = 0;

				if (toAdd == 1)
					toAdd = 0;

				midDst[i] = sum;
			}

			for (int i = size - 1; i >= 0; i--)
				if (low[i] > high[i]) {
					keepRunning = false;
					break;
				}
		}
	}

	// floor(cube root), since high^3 <= x < (high+1)^3
	// high is already loaded into the output at this point
}

void _internalBigintFastSqrt(const uint64_t* x, uint64_t* y, int size) {
	// Determine approximate bit length of x
	int highestWord = size - 1;
	while (highestWord > 0 && x[highestWord] == 0)
		--highestWord;
	uint64_t hi = x[highestWord];
	int bitlen = countBits(hi) + highestWord * 64;

	if (bitlen == 0) {
		bzero(y, sizeof(uint64_t) * size);
		return;
	}

	// -------- Solidity-style initial guess --------
	int rShift = 0;
	int xxBits = bitlen;

	if (xxBits >= 128 + 128) { xxBits -= 128; rShift += 64; }
	if (xxBits >= 64 + 64)   { xxBits -= 64;  rShift += 32; }
	if (xxBits >= 32 + 32)   { xxBits -= 32;  rShift += 16; }
	if (xxBits >= 16 + 16)   { xxBits -= 16;  rShift += 8; }
	if (xxBits >= 8 + 8)     { xxBits -= 8;   rShift += 4; }
	if (xxBits >= 4 + 4)     { xxBits -= 4;   rShift += 2; }
	if (xxBits >= 3 + 3)     { rShift += 1; }

	// r = 1 << rShift
	uint64_t r[size];
	memset(r, 0, size * sizeof(uint64_t));
	int wShift = rShift / 64;
	int bShift = rShift % 64;
	if (wShift < size) {
		r[wShift] = 1ULL << bShift;
	}

	// -------- Newton refinement: r = (r + x / r) >> 1 repeated 7 times --------
	uint64_t tmp[size], div[size];
	for (int iter = 0; iter < 7; iter++) {
		// div = x / r
		memcpy(tmp, x, sizeof(uint64_t) * size);
		_internalBigintFastDiv(tmp, r, div, size);

		// tmp = r + div
		uint64_t carry = 0;
		for (int i = 0; i < size; i++) {
			uint64_t sum = r[i] + div[i] + carry;
			carry = (sum < r[i] || (carry && sum == r[i]));
			tmp[i] = sum;
		}

		// r = tmp >> 1
		uint64_t prev = 0;
		for (int i = size - 1; i >= 0; i--) {
			uint64_t cur = tmp[i];
			r[i] = (cur >> 1) | (prev << 63);
			prev = cur;
		}
	}

	// -------- Final correction: take min(r, x / r) --------
	memcpy(tmp, x, sizeof(uint64_t) * size);
	_internalBigintFastDiv(tmp, r, div, size);

	for (int i = size - 1; i >= 0; i--) {
		if (r[i] < div[i]) {
			memcpy(y, r, size * sizeof(uint64_t));
			return;
		} else if (r[i] > div[i])
			break;
	}

	memcpy(y, div, size * sizeof(uint64_t));
}


/*void _internalBigintFastSmallRoot(const uint64_t* x, uint64_t* y, int size, int root) {
	if (root == 1)
		memcpy(y, x, size * sizeof(uint64_t));
	else if (root <= 0)
		memset(y, 0xFF, size * sizeof(uint64_t));

	uint16_t inputBits = 0;
	for (int i = size - 1; i >= 0; i--)
		if (x[i] != 0) {
			inputBits = countBits(x[i]) + 64 * i;
			break;
		}
	if (inputBits <= 1) {
		// root of 1 is always 1
		memcpy(y, x, size * sizeof(uint64_t));
		return;
	}

	// initial guess: r = 1 << (bitlen / n + 1)
	uint64_t r[size];
	bzero(r, size * sizeof(uint64_t));
	int rShift = inputBits / root + 1;
	int wShift = rShift / 64;
	int bShift = rShift % 64;
	r[wShift] = 1ULL << bShift;

	// --- Newton-Raphson iterations: r <- ((n-1)*r + x / r^(n-1)) / n ---
	uint64_t tmp[size];
	uint64_t div[size];
	uint64_t pow[size];
	for (int iter = 0; iter < 8; ++iter) {
		// pow = r^(n-1)
		_internalBigintFastPow(r, pow, size, root-1);

		// div = x / pow
		memcpy(tmp, x, size * sizeof(uint64_t));
		_internalBigintFastDiv(tmp, pow, div, size);

		// tmp = (n-1)*r + div
		uint64_t carry = 0;
		for (int i = 0; i < size; ++i) {
			__uint128_t sum = (__uint128_t)(root - 1) * r[i] + div[i] + carry;
			tmp[i] = (uint64_t)sum;
			carry = (uint64_t)(sum >> 64);
		}

		// r = tmp / n
		__uint128_t rem = 0;
		for (int i = size - 1; i >= 0; --i) {
			__uint128_t v = (rem << 64) | tmp[i];
			uint64_t q = (uint64_t)(v / root);
			rem = v - (__uint128_t)q * root;
			r[i] = q;
		}
	}

	// final result
	memcpy(y, r, size * sizeof(uint64_t));
}*/


void _internalBigintFastSmallRoot(const uint64_t* x, uint64_t* y, int size, int root) {
	if (root == 1)
		memcpy(y, x, size * sizeof(uint64_t));
	else if (root <= 0)
		memset(y, 0xFF, size * sizeof(uint64_t));

	uint16_t inputBits = 0;
	for (int i = size - 1; i >= 0; i--)
		if (x[i] != 0) {
			inputBits = countBits(x[i]) + 64 * i;
			break;
		}
	if (inputBits <= 1) {
		// root of 1 is always 1
		memcpy(y, x, size * sizeof(uint64_t));
		return;
	}

	// Compute initial guess R
	uint64_t* r = y;
	bzero(r, size * sizeof(uint64_t));
	uint16_t shift = inputBits / root + 1;
	uint16_t wordShift = shift / 64;
	uint16_t bitShift = shift & 63;
	r[wordShift] = 1UL << bitShift;

	uint64_t longRoot[size];
	bzero(longRoot, size * sizeof(uint64_t));
	longRoot[0] = root;

	uint64_t pow[size];
	uint64_t x_div_pow[size];
	uint64_t tmp[size];
	uint64_t r_next[size];
	for (int _ = 0; _ < 8; _++) {
		_internalBigintFastPow(r, pow, size, root - 1);
		bool isZero = true;
		for (int j = 0; j < size; j++)
			if (pow[j] != 0) {
				isZero = false;
				break;
			}
		if (isZero)
			break;

		// x_div_pow = x / (r**(root - 1))
		memcpy(tmp, x, size * sizeof(uint64_t));
		bzero(x_div_pow, size * sizeof(uint64_t));
		_internalBigintFastDiv(tmp, pow, x_div_pow, size);

		// x_div_pow := r * (root - 1) + x_div_pow
		uint64_t carry = 0;
		for (int i = 0; i < size; ++i) {
			__uint128_t a = r[i];
			__uint128_t prod = a * (root - 1) + x_div_pow[i] + carry;
			x_div_pow[i] = (uint64_t)prod;
			carry = (uint64_t)(prod >> 64);
		}

		bzero(r_next, size * sizeof(uint64_t));
		_internalBigintFastDiv(x_div_pow, longRoot, r_next, size);

		bool isNextRGreater = false;
		for (int i = size - 1; i >= 0; i--) {
			if (r_next[i] > r[i]) {
				isNextRGreater = true;
				break;
			} else if (r_next[i] < r[i])
				break;
		}

		if (isNextRGreater)
			break;

		memcpy(r, r_next, sizeof(uint64_t) * size);
	}

	// No need to memcpy r -> y because r refers to the same memory
}


double _internalBigintConvertToDouble(uint16_t bits, int N, const uint64_t* chunks) {
	if (bits == 0)
		return 0.0;

	union {
		uint64_t raw;
		double val;
	} out;

	if (bits < 53) {
		out.raw = chunks[0] << (53 - bits);

		// Clear the implicit 53rd bit to make room for the exponent
		out.raw ^= (uint64_t)1 << 52;

		// Set the exponent
		uint64_t exponent = 1023 + bits - 1;
		out.raw |= exponent << 52;

	} else {
		uint16_t startBit = bits - 53;
		uint16_t shift1 = startBit & 63;
		uint16_t shift2 = 64 - shift1;

		// Set the significand
		out.raw = chunks[startBit / 64] >> shift1;
		if ((startBit + 63) / 64 < N)
			out.raw |= chunks[(startBit + 63) / 64] << shift2;

		// Clear the implicit 53rd bit to make room for the exponent
		out.raw ^= (uint64_t)1 << 52;

		// Set the exponent
		uint64_t exponent = 1023 + bits - 1;
		out.raw |= exponent << 52;
	}

	return out.val;
}

void _internalBigintConvertFromDouble(double value, int N, uint64_t* chunks) {
	if (value < 0.5 && value >= 0.0) {
		bzero(chunks, N * sizeof(uint64_t));
		return;
	}

	union {
		uint64_t raw;
		double val;
	} out;
	out.val = value;

	bool isPositive = (out.raw >> 63) == 0;
	out.raw &= 0x7fffffffffffffffUL;

	uint64_t fraction = (out.raw & 0xfffffffffffffUL) | 0x10000000000000;
	int exponent = (int)(out.raw >> 52) - 1023;
	int shift = exponent - 52;

	if (shift < 0) {
		fraction = fraction >> -shift;
		shift = 0;
	}

	int wordShift = shift / 64;
	int bitShift1 = shift & 63;
	int bitShift2 = 64 - bitShift1;

	if (isPositive) {
		bzero(chunks, 8*N);

		chunks[wordShift] = fraction << bitShift1;
		if (bitShift1 != 0 && wordShift + 1 < N)
			chunks[wordShift + 1] = fraction >> bitShift2;
	} else {
		memset(chunks, 0xff, 8*N);
	}
}


void _internalBigintFastMulDouble(const uint64_t* a, double b, uint64_t* y, int size) {
	bzero(y, size * sizeof(uint64_t));

	if (b == 0.0)
		return;

	union {
		uint64_t raw;
		double val;
	} out;
	out.val = b;

	out.raw &= 0x7fffffffffffffffUL;

	uint64_t fraction = (out.raw & 0xfffffffffffffUL) | 0x10000000000000;
	int exponent = (int)(out.raw >> 52) - 1023;
	int shift = exponent - 52;
	int wordShift = shift / 64;
	int bitShift1 = abs(shift) & 63;
	int bitShift2 = 64 - bitShift1;

	if (abs(wordShift) >= size)
		return;

	// First compute multiplication by the mantissa
	uint64_t carry = 0;
	for (int i = wordShift; i < size; ++i) {
		__uint128_t word = (i - wordShift >= size) ? 0 : a[i - wordShift];
		__uint128_t intermediate = (word * (__uint128_t)fraction) + carry;
		if (i >= 0)
			y[i] = intermediate & 0xFFFFFFFFFFFFFFFFUL;
		carry = intermediate >> 64;
	}

	if (bitShift1 == 0)
		return;

	if (shift >= 0) {
		// Shift left
		for (int i = size - 1; i >= 0; i--) {
			y[i] = y[i] << bitShift1;
			if (bitShift1 && i > 0)
				y[i] |= y[i - 1] >> bitShift2;
		}
	} else {
		// Shift right
		for (int i = 0; i < size; i++) {
			y[i] = y[i] >> bitShift1;
			if (bitShift1 && i + 1 < size)
				y[i] |= y[i + 1] << bitShift2;
		}
	}
}
