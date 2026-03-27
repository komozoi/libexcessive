/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2023-04-04
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


#ifndef EXCESSIVE_MOREMATH_H
#define EXCESSIVE_MOREMATH_H


#include <cstdint>

// Measuring correlation
double correlation(const double* sequence1, const double* sequence2, int length);
void slidingCorrelation(const double* first, const double* second, double* out, int length);

// Polyfits & fit errors
void polyfit(const double* x, const double* y, int inputLength, double* out, int deg);
double polyval(const double* polyfit, int deg, double x);
void polyval(const double* polyfit, int deg, double* x, double* y, int length);
void multivariateLinearFit(int n, int m, double** x, const double* y, double *coefficients);
void fitSineFunction(const double* x, const double* y, int length, double& phase, double& amplitude);
double mse(const double* a, const double* b, int length);
float mse(const float* a, const float* b, int length);
//double calculateRSquared(const double* a, const double* b, int length);
double calculateMAPE(const double* a, const double* b, int length);
double fitMatchCoef(const double* a, const double* b, int length);

// Vector math utilities
void square(const double* x, double* xx, int length);
void sqrt(const double* xx, double* x, int length);
void sub(const double* a, const double* b, double* y, int n);

// Statistics
double mean(const double* vals, int length);
double nanmedian(const double* vals, int length);
double variance(const double* vals, int length);
double variance(const double* vals, int length, double mean);
double stddev(const double* vals, int length);
double stddev(const double* vals, int length, double mean);
double norminv(double p);

// Other
double min(const double* arr, int length);
double max(const double* arr, int length);
int indexOfMaximum(const double* arr, int length);
double interp(double x, const double* xp, const double* fp, int length);

// Data manipulation
void sort(double* arr, int length);
// This will sort arr1, and move the elements of arr2 in the same pattern.
void sortTogether(double* arr1, double* arr2, int length);
int argmaxf(float* arr, int length);

// Working with frequencies and oscillations
double frequencyOf(double* samples, int length, double samplingRate=1);

// Matrices
void addMatf(float* A, float* B, float* C, int n);
void subMatf(float* A, float* B, float* C, int n);
void mulMatf(const float* m1, const float* m2, float* out, int n, int m, int p);
void fastMulMatf(const float* m1, const float* m2, float* out, int n, int m, int p);
void mulScalarMatf(const float* mat, float scalar, float* out, int n);
void mulElemwiseMatf(const float* m1, const float* m2, float* out, int n);
void divElemwiseMatf(const float* m1, const float* m2, float* out, int n);
// This function is much faster for matrices 128x128 or bigger, but is not as accurate.
// For values in the matrix being less than, say, 10, expect the error to be +/- 0.005%.
// The error goes up with bigger values in the matrix, so expect a much worse error if your matrix has
// values 100 or bigger.  In such a case the error can approach or exceed 1%.
void mulMatStrassenRecursive(const float* A, const float* B, float* C, int n);


static inline uint8_t countSetBits(uint64_t value) {
	return __builtin_popcountll(value);
	/*
	value = ((value & 0xAAAAAAAAAAAAAAAAUL) >> 1) + (value & 0x5555555555555555UL);
	value = ((value & 0xCCCCCCCCCCCCCCCCUL) >> 2) + (value & 0x3333333333333333UL);
	value = ((value & 0xF0F0F0F0F0F0F0F0UL) >> 4) + (value & 0x0F0F0F0F0F0F0F0FUL);
	value = ((value & 0xFF00FF00FF00FF00UL) >> 8) + (value & 0x00FF00FF00FF00FFUL);
	value = ((value & 0xFFFF0000FFFF0000UL) >> 16) + (value & 0x0000FFFF0000FFFFUL);
	value = ((value & 0xFFFFFFFF00000000UL) >> 32) + (value & 0x00000000FFFFFFFFUL);

	return value;*/
}


static inline uint8_t countBits(uint64_t value) {
	if (value == 0) return 0;
	return 64 - __builtin_clzll(value);
}


#ifdef EXCESSIVE_ARRAYLIST_H
void getMaxIndexes(ArrayList<int>& indexes, const double* data, int length, double cutoff);
void getMaxIndexes(ArrayList<int>& indexes, const double* data, int length, double cutoff, double scale, double offset);
void getPeriods(ArrayList<double>& periodIndexes, ArrayList<double>& periods, double* x_vals, double* y_vals, int length);
#endif


#endif //EXCESSIVE_MOREMATH_H
