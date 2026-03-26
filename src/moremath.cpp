//
// Created by komozoi on 4/4/23.
//

#include "math.h"
#include "moremath.h"
#include "ds/ArrayList.h"

// Gonna generate most of this with ChatGPT.  Let's see how it goes.  Will need to make some modifications, so in the
// reality this will be human-assisted AI generated code.

double calculateCorrelation(double sum_x, double sum_y, double sum_xy, double sum_x2, double sum_y2, int n) {
	double denominator = sqrt((sum_x2 - (sum_x * sum_x) / n) * (sum_y2 - (sum_y * sum_y) / n));

	// Prevent division by zero and negative values under the square root
	if (denominator == 0) {
		return 0.0;
	}

	return (sum_xy - (sum_x * sum_y) / n) / denominator;
}

double correlation(const double* sequence1, const double* sequence2, int length) {

	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum12 = 0.0;
	double sum1sq = 0.0;
	double sum2sq = 0.0;

	double n = 0;
	for (int i = 0; i < length; i++) {
		if (!(isnan(sequence1[i]) || isnan(sequence2[i]))) {
			sum1 += sequence1[i];
			sum2 += sequence2[i];
			sum12 += sequence1[i] * sequence2[i];
			sum1sq += sequence1[i] * sequence1[i];
			sum2sq += sequence2[i] * sequence2[i];
			n++;
		}
	}
	return calculateCorrelation(sum1, sum2, sum12, sum1sq, sum2sq, n);
}

/*void slidingCorrelation(const double* first, const double* second, double* out, int length) {
	double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;

	if (length <= 1)
		return;

	// Initialize sums for the first window
	for (int i = 0; i < length; i++) {
		sum_x += first[i];
		sum_y += second[i];
		sum_xy += first[i] * second[i];
		sum_x2 += first[i] * first[i];
		sum_y2 += second[i] * second[i];
	}

	// Calculate correlation for the first window
	out[0] = calculateCorrelation(sum_x, sum_y, sum_xy, sum_x2, sum_y2, length);

	// Slide the window and update the correlation
	for (int i = 0; i < length - 2; i++) {
		// Each time, we remove the last element from first and the first element from last.
		sum_x -= first[length - i - 1];
		sum_y -= second[i];
		sum_x2 -= first[length - i - 1] * first[length - i - 1];
		sum_y2 -= second[i] * second[i];

		sum_xy = 0;
		for (int j = 0; j < length - i - 1; j++)
			sum_xy += first[j] * second[j + i + 1];
		// The following does not work, but if we could make something like this work it would be faster.
		//sum_xy -= first[i - 1] * second[i - 1];
		//sum_xy += first[length - i - 1] * second[length - 1];

		out[i + 1] = calculateCorrelation(sum_x, sum_y, sum_xy, sum_x2, sum_y2, length - i - 1);
	}
}*/

void slidingCorrelation(const double* first, const double* second, double* out, int length) {
	if (length <= 1)
		return;
	for (int i = 0; i < length - 1; i++) {
		out[i] = correlation(first, &second[i], length - i);
	}
}

/*void polyfit(const double* x, const double* y, int inputLength, double* out, int deg) {
	int n = inputLength;
	int i, j, k;
	double X[2 * deg + 1];
	double Y[deg + 1];
	double B[deg + 1];

	for (i = 0; i < 2 * deg + 1; i++) {
		X[i] = 0.0;
		for (j = 0; j < n; j++)
			X[i] += pow(x[j], i);
	}

	for (i = 0; i < deg + 1; i++) {
		Y[i] = 0.0;
		for (j = 0; j < n; j++)
			Y[i] += pow(x[j], i) * y[j];
	}

	for (i = 0; i < deg + 1; i++) {
		B[i] = 0.0;
		for (j = 0; j < deg + 1; j++)
			B[i] += X[i + j] * Y[j];
	}

	double* A[deg + 1];
	for (i = 0; i < deg + 1; i++) {
		A[i] = new double[deg + 1];
		for (j = 0; j < deg + 1; j++)
			A[i][j] = X[i + j];
	}

	for (k = 0; k < deg + 1; k++) {
		for (i = k + 1; i < deg + 1; i++) {
			double f = A[i][k] / A[k][k];
			for (j = k + 1; j < deg + 1; j++) {
				A[i][j] -= f * A[k][j];
			}
			B[i] -= f * B[k];
		}
	}

	for (k = deg; k >= 0; k--) {
		out[k] = B[k];
		for (j = k + 1; j < deg + 1; j++) {
			out[k] -= A[k][j] * out[j];
		}
		out[k] /= A[k][k];
	}
}*/

void polyfit(const double* x, const double* y, int inputLength, double* out, int deg) {
	int i, j, k;
	double X[2 * deg + 1];
	double Y[deg + 1];
	//double B[deg + 1];

	for (i = 0; i < 2 * deg + 1; i++) {
		X[i] = 0.0;
		for (j = 0; j < inputLength; j++) {
			if (!(isnan(x[j]) || isnan(y[j])))
				X[i] += pow(x[j], i);
		}
	}

	for (i = 0; i <= deg; i++) {
		Y[i] = 0.0;
		for (j = 0; j < inputLength; j++) {
			if (!(isnan(x[j]) || isnan(y[j])))
				Y[i] += pow(x[j], i) * y[j];
		}
	}

	double A[deg + 1][deg + 1];
	for (i = 0; i <= deg; i++) {
		for (j = 0; j <= deg; j++) {
			A[i][j] = X[i+j];
		}
	}

	// Gaussian elimination
	for (k = 0; k < deg; k++) {
		for (i = k + 1; i <= deg; i++) {
			double factor = A[i][k] / A[k][k];
			for (j = k + 1; j <= deg; j++) {
				A[i][j] -= factor * A[k][j];
			}
			Y[i] -= factor * Y[k];
		}
	}

	// Back-substitution
	for (i = deg; i >= 0; i--) {
		double s = Y[i];
		for (j = i + 1; j <= deg; j++) {
			s -= A[i][j] * out[deg - j];
		}
		out[deg - i] = s / A[i][i];
	}
}


// I actually wrote this one
double polyval(const double* polyfit, int deg, double x) {
	double accumulator = 0;
	double multiplier = 1;
	for (int i = deg; i >= 0; i--) {
		accumulator += multiplier * polyfit[i];
		multiplier *= x;
	}
	return accumulator;
}


void polyval(const double* polyfit, int deg, double* x, double* y, int length) {
	for (int i = 0; i < length; i++) {
		y[i] = polyval(polyfit, deg, x[i]);
	}
}

void multivariateLinearFit(int n, int m, double** x, const double* y, double *coefficients) {
	int i, j, k;

	double x_sum[m + 1];
	double x_product[m + 1][m + 1];
	double y_product[m + 1];

	// Initialize sums and products
	for (i = 0; i <= m; ++i) {
		x_sum[i] = 0;
		y_product[i] = 0;
		for (j = 0; j <= m; ++j) {
			x_product[i][j] = 0;
		}
	}

	// Calculate x_sum and x_product
	int valid_data_points = 0;
	for (i = 0; i < n; ++i) {
		int skip = 0;
		if (isnan(y[i])) skip = 1;
		for (j = 0; j < m; ++j) {
			if (isnan(x[j][i])) {
				skip = 1;
				break;
			}
		}
		if (skip) continue;

		valid_data_points++;

		for (j = 0; j <= m; ++j) {
			for (k = 0; k <= m; ++k) {
				x_product[j][k] += (j > 0 ? x[j - 1][i] : 1) * (k > 0 ? x[k - 1][i] : 1);
			}
		}

		// Calculate y_product
		y_product[0] += y[i];
		for (j = 1; j <= m; ++j) {
			y_product[j] += x[j - 1][i] * y[i];
		}
	}

	// Update x_sum
	x_sum[0] = valid_data_points;
	for (j = 1; j <= m; ++j) {
		x_sum[j] = 0;
		for (i = 0; i < n; ++i) {
			if (!isnan(x[j - 1][i])) {
				x_sum[j] += x[j - 1][i];
			}
		}
	}

	// Gaussian elimination with partial pivoting
	for (j = 0; j <= m; ++j) {
		// Find the pivot row
		int pivot_row = j;
		for (i = j + 1; i <= m; ++i) {
			if (fabs(x_product[i][j]) > fabs(x_product[pivot_row][j])) {
				pivot_row = i;
			}
		}

		// Swap the pivot row with the current row
		if (pivot_row != j) {
			for (k = j; k <= m; ++k) {
				double tmp = x_product[j][k];
				x_product[j][k] = x_product[pivot_row][k];
				x_product[pivot_row][k] = tmp;
			}
			double tmp = y_product[j];
			y_product[j] = y_product[pivot_row];
			y_product[pivot_row] = tmp;
		}

		// Check if the pivot element is very close to zero
		if (fabs(x_product[j][j]) < 0.0000001) {
			continue;
		}

		// Perform elimination
		for (i = j + 1; i <= m; ++i) {
			double factor = x_product[i][j] / x_product[j][j];
			for (k = j + 1; k <= m; ++k) {
				x_product[i][k] -= factor * x_product[j][k];
			}
			y_product[i] -= factor * y_product[j];
		}
	}

	// Back-substitution
	for (j = m; j >= 0; --j) {
		if (fabs(x_product[j][j]) < 0.000001) {
			coefficients[j] = 0;
			continue;
		}

		coefficients[j] = y_product[j];
		for (i = j + 1; i <= m; ++i) {
			coefficients[j] -= x_product[j][i] * coefficients[i];
		}
		coefficients[j] /= x_product[j][j];
	}
}


// Another GPT answer, with some changes.  It's shitty but it works.
void fitSineFunction(const double* x, const double* y, int length, double& phase, double& amplitude) {
	// Allocate memory for the intermediate variables
	double AS = 0.0, BS = 0.0, AC = 0.0, BC = 0.0;

	// Compute the coefficients
	for (int i = 0; i < length; i++) {
		double c = cos(x[i]);
		double s = sin(x[i]);
		AS += y[i] * c;
		BS += y[i] * s;
		AC += c * c;
		BC += s * c;
	}

	// Compute the amplitude and phase
	phase = atan2(AS * BC - AC * BS, AS * AC + BS * BC) + M_PI_2;
	double sum = 0;
	int samples = 0;
	for (int i = 0; i < length; i++) {
		double s = sin(x[i] + phase);
		if (abs(s - 0) > 0.00001) {
			sum += y[i] / s;
			samples++;
		}
	}
	amplitude = sum / samples;
}

double mse(const double* a, const double* b, int length) {
	double sum = 0.0;
	for (int i = 0; i < length; i++) {
		double diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum / length;
}

float mse(const float* a, const float* b, int length) {
	float sum = 0.0;
	for (int i = 0; i < length; i++) {
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum / (float)length;
}

/*double calculateRSquared(const double* a, const double* b, int length) {
	if (length <= 1)
		return 0.0;

	double mean_a = mean(a, length);

	double ss_res = 0, ss_tot = 0;
	for (int i = 0; i < length; i++) {
		double a_diff = a[i] - mean_a;
		double diff = b[i] - a[i];
		ss_res += diff * diff;
		ss_tot += a_diff * a_diff;
	}

	if (ss_tot == 0)
		return 0.0;

	return 1 - (ss_res / ss_tot);
}*/

double calculateMAPE(const double* a, const double* b, int length) {
	double sum_abs_percent_diff = 0.0;
	for (int i = 0; i < length; i++) {
		if (a[i] == 0) {
			printf("Division by zero encountered in the dataset!\n");
		}

		double abs_percent_diff = std::abs((a[i] - b[i]) / a[i]) * 100;
		sum_abs_percent_diff += abs_percent_diff;
	}

	return sum_abs_percent_diff / length;
}

double fitMatchCoef(const double* a, const double* b, int length) {
	double sum_abs_percent_diff = 0.0;
	int count = 0;
	for (int i = 0; i < length; i++) {
		if (isnan(a[i] + b[i]))
			continue;
		count++;
		double denominator = std::abs(a[i]) + std::abs(b[i]);
		if (denominator == 0)
			continue;

		double abs_percent_diff = std::abs((a[i] - b[i]) / denominator);
		sum_abs_percent_diff += abs_percent_diff;
	}

	return 1 - sum_abs_percent_diff / count;
}

void square(const double* x, double* xx, int length) {
	for (int i = 0; i < length; i++) {
		xx[i] = x[i] * x[i];
	}
}

void sqrt(const double* xx, double* x, int length) {
	for (int i = 0; i < length; i++) {
		x[i] = sqrt(xx[i]);
	}
}

void sub(const double* a, const double* b, double* y, int n) {
	for (int i = 0; i < n; i++) {
		y[i] = a[i] - b[i];
	}
}

double mean(const double* vals, int length) {
	double out = 0;
	for (int i = 0; i < length; i++)
		out += vals[i];
	return out / length;
}

// Define the nanmedian function
double nanmedian(const double* vals, int length) {
	// Create a copy of the input array to sort
	double sorted[length];
	int newLength = 0;
	for (int i = 0; i < length; i++) {
		if (isfinite(vals[i]))
			sorted[newLength++] = vals[i];
	}
	if (newLength == 0)
		return NAN;

	// Sort the copy of the array
	sort(sorted, newLength);

	// Find the median value
	int middle = newLength / 2;
	if (newLength % 2 == 0) {
		// If the length is even, take the average of the middle two values
		double a = sorted[middle - 1];
		double b = sorted[middle];
		return (a + b) / 2;
	} else {
		// If the length is odd, take the middle value
		return sorted[middle];
	}
}

double variance(const double* vals, int length) {
	double sum = 0;
	double sum_sq = 0;
	for (int i = 0; i < length; i++) {
		sum += vals[i];
		sum_sq += vals[i] * vals[i];
	}
	double mean = sum / length;
	double variance = sum_sq / length - mean * mean;
	return variance;
}

double variance(const double* vals, int length, double mean) {
	double variance = 0;
	for (int i = 0; i < length; i++) {
		double diff = vals[i] - mean;
		variance += diff * diff;
	}
	variance /= length;
	return variance;
}

double stddev(const double* vals, int length) {
	return sqrt(variance(vals, length));
}

double stddev(const double* vals, int length, double mean) {
	return sqrt(variance(vals, length, mean));
}

// Vruka ChatGPT.
// ChatGPT: "Note that the norminv function is an approximation of the inverse standard normal distribution.
//           It uses coefficients derived from the work of Peter J. Acklam."
double norminv(double p) {
	const double a1 = -39.6968302866538;
	const double a2 = 220.946098424521;
	const double a3 = -275.928510446969;
	const double a4 = 138.357751867269;
	const double a5 = -30.6647980661472;
	const double a6 = 2.50662827745924;

	const double b1 = -54.4760987982241;
	const double b2 = 161.585836858041;
	const double b3 = -155.698979859887;
	const double b4 = 66.8013118877197;
	const double b5 = -13.2806815528857;

	const double c1 = -0.00778489400243029;
	const double c2 = -0.322396458041136;
	const double c3 = -2.40075827716184;
	const double c4 = -2.54973253934373;
	const double c5 = 4.37466414146497;
	const double c6 = 2.93816398269878;

	const double d1 = 0.00778469570904146;
	const double d2 = 0.32246712907004;
	const double d3 = 2.445134137143;
	const double d4 = 3.75440866190742;

	const double pLow = 0.02425;
	const double pHigh = 1 - pLow;
	double q, r;

	if (p < 0 || p > 1) {
		printf("Error: Probability argument must be between 0 and 1.\n");
		return 0.0;
	}

	if (p < pLow) {
		q = sqrt(-2 * log(p));
		return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
			   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
	}

	if (p <= pHigh) {
		q = p - 0.5;
		r = q * q;
		return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
			   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
	}

	q = sqrt(-2 * log(1 - p));
	return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
		   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
}


double min(const double* arr, int length) {
	double least = INFINITY;
	for (int i = 0; i < length; i++) {
		if (least > arr[i])
			least = arr[i];
	}
	return least;
}

double max(const double* arr, int length) {
	double least = -(double)INFINITY;
	for (int i = 0; i < length; i++) {
		if (least < arr[i])
			least = arr[i];
	}
	return least;
}

int indexOfMaximum(const double* arr, int length) {
	double least = arr[0];
	int iMax = 0;
	for (int i = 1; i < length; i++) {
		if (least < arr[i]) {
			least = arr[i];
			iMax = i;
		}
	}
	return iMax;
}

// Generated by ChatGPT
double interp(double x, const double* xp, const double* fp, int length) {
	if (x < xp[0]) {
		double slope = (fp[1] - fp[0]) / (xp[1] - xp[0]);
		return fp[0] + slope * (x - xp[0]);
	} else if (x > xp[length - 1]) {
		double slope = (fp[length - 1] - fp[length - 2]) / (xp[length - 1] - xp[length - 2]);
		return fp[length - 1] + slope * (x - xp[length - 1]);
	}
	int i;
	for (i = 0; i < length; i++) {
		if (xp[i] >= x) {
			break;
		}
	}
	if (xp[i] == x) {
		return fp[i];
	}
	double slope = (fp[i] - fp[i - 1]) / (xp[i] - xp[i - 1]);
	return fp[i - 1] + slope * (x - xp[i - 1]);
}


void sort(double* arr, int length) {
	for (int i = 0; i < length - 1; i++) {
		for (int j = i + 1; j < length; j++) {
			if (arr[i] > arr[j]) {
				double temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
			}
		}
	}
}


void sortTogether(double* arr1, double* arr2, int length) {
	for (int i = 0; i < length - 1; i++) {
		for (int j = i + 1; j < length; j++) {
			if (arr1[i] > arr1[j]) {
				double temp = arr1[i];
				arr1[i] = arr1[j];
				arr1[j] = temp;
				temp = arr2[i];
				arr2[i] = arr2[j];
				arr2[j] = temp;
			}
		}
	}
}

int argmaxf(float* arr, int length) {
	int m = 0;
	float v = -INFINITY;
	for (int i = 0; i < length; i++)
		if (arr[i] > v) {
			v = arr[i];
			m = i;
		}
	return m;
}


void getMaxIndexes(ArrayList<int>& indexes, const double* data, int length, double cutoff) {
	int lastMaxIdx = -1;
	double lastMax = 0;
	for (int i = 0; i < length; i++) {
		if (lastMaxIdx != -1) {
			if (data[i] > lastMax) {
				lastMaxIdx = i;
				lastMax = data[i];
			} else if (data[i] < cutoff) {
				indexes.add(lastMaxIdx);
				lastMaxIdx = -1;
			}
		} else {
			if (data[i] > cutoff) {
				lastMaxIdx = i;
				lastMax = data[i];
			}
		}
	}
}

void getMaxIndexes(ArrayList<int>& indexes, const double* data, int length, double cutoff, double scale, double offset) {
	int lastMaxIdx = -1;
	double lastMax = 0;
	for (int i = 0; i < length; i++) {
		double point = data[i] * scale + offset;
		if (lastMaxIdx != -1) {
			if (point > lastMax) {
				lastMaxIdx = i;
				lastMax = point;
			} else if (point < cutoff) {
				indexes.add(lastMaxIdx);
				lastMaxIdx = -1;
			}
		} else {
			if (point > cutoff) {
				lastMaxIdx = i;
				lastMax = point;
			}
		}
	}
}


// This is designed for cases where FFTs don't work well.
// For example, getting the frequency at a specific point
// in time where the frequency is changing.  The tradeoff
// is accuracy - this function is much more susceptible to
// noise.
double frequencyOf(double* samples, int length, double samplingRate) {
	double sampleMean = mean(samples, length);
	double variance = ::variance(samples, length, sampleMean);
	ArrayList<int> maxIndexes, minIndexes;
	getMaxIndexes(maxIndexes, samples, length, variance, 1, -sampleMean);
	getMaxIndexes(minIndexes, samples, length, -variance, -1, sampleMean);

	ArrayList<double> periods;
	int i = 0;
	int j = 0;
	while (i < maxIndexes.size() && j < minIndexes.size()) {
		if (maxIndexes.get(i) > minIndexes.get(j)) {
			/* Waveform:
			**     /-\
            **    |
			** \_/
			*/
			while (minIndexes.size() > j + 1 && maxIndexes.get(i) > minIndexes.get(j + 1))
				j++;

			periods.add(maxIndexes.get(i) - minIndexes.get(j));
			j++;

		} else if (maxIndexes.get(i) < minIndexes.get(j)) {
			/* Waveform:
			** /-\
            **    |
			**     \_/
			*/
			while (maxIndexes.size() > i + 1 && maxIndexes.get(i + 1) < minIndexes.get(j))
				i++;

			periods.add(minIndexes.get(j) - maxIndexes.get(i));
			i++;
		}
	}

	return samplingRate / nanmedian(periods.getMemory(), periods.size());
}


// This is designed for cases where FFTs don't work well.
// For example, getting the frequency at a specific point
// in time where the frequency is changing.  The tradeoff
// is accuracy - this function is much more susceptible to
// noise.
void getPeriods(ArrayList<double>& periodIndexes, ArrayList<double>& periods, double* x_vals, double* y_vals, int length) {
	double sampleMean = mean(y_vals, length);
	double std_dev = stddev(y_vals, length, sampleMean);
	ArrayList<int> maxIndexes, minIndexes;
	getMaxIndexes(maxIndexes, y_vals, length, std_dev, 1, -sampleMean);
	getMaxIndexes(minIndexes, y_vals, length, -std_dev, -1, sampleMean);

	int i = 0;
	int j = 0;
	while (i < maxIndexes.size() && j < minIndexes.size()) {
		if (maxIndexes.get(i) > minIndexes.get(j)) {
			/* Waveform:
			**     /-\
            **    |
			** \_/
			*/
			while (minIndexes.size() > j + 1 && maxIndexes.get(i) > minIndexes.get(j + 1))
				j++;

			periods.add(x_vals[maxIndexes.get(i)] - x_vals[minIndexes.get(j)]);
			periodIndexes.add((x_vals[minIndexes.get(j)] + x_vals[maxIndexes.get(i)]) / 2);
			j++;
		} else if (maxIndexes.get(i) < minIndexes.get(j)) {
			/* Waveform:
			** /-\
            **    |
			**     \_/
			*/
			while (maxIndexes.size() > i + 1 && maxIndexes.get(i + 1) < minIndexes.get(j))
				i++;

			periods.add(x_vals[minIndexes.get(j)] - x_vals[maxIndexes.get(i)]);
			periodIndexes.add((x_vals[minIndexes.get(j)] + x_vals[maxIndexes.get(i)]) / 2);
			i++;
		}
	}
}


void addMatf(float* A, float* B, float* C, int n) {
	for (int i = 0; i < n; i++) {
		C[i] = A[i] + B[i];
	}
}

void subMatf(float* A, float* B, float* C, int n) {
	for (int i = 0; i < n; i++) {
		C[i] = A[i] - B[i];
	}
}

void mulMatf(const float* m1, const float* m2, float* out, int n, int m, int p) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			out[i * p + j] = 0;
			for (int k = 0; k < m; k++) {
				out[i * p + j] += m1[i * m + k] * m2[k * p + j];
			}
		}
	}
}

void fastMulMatf(const float* m1, const float* m2, float* out, int n, int m, int p) {
	if (n == m && n == p) {
		mulMatStrassenRecursive(m1, m2, out, n);
	} else {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < p; j++) {
				out[i * p + j] = 0;
				for (int k = 0; k < m; k++) {
					out[i * p + j] += m1[i * m + k] * m2[k * p + j];
				}
			}
		}
	}
}

// Shockingly, GPT4 nailed this pretty well the first time.  I have still optimized it since, however, and gotten
// incredible speed improvements.
void mulMatStrassenRecursive(const float* A, const float* B, float* C, int n) {
	if (n <= 64) {
		mulMatf(A, B, C, n, n, n);
		return;
	}

	int new_size = n >> 1;
	int new_size_len = new_size * new_size;
	float* memory = new float[new_size_len * 29];
	float* a11 = &memory[new_size_len * 0];
	float* a12 = &memory[new_size_len * 1];
	float* a21 = &memory[new_size_len * 2];
	float* a22 = &memory[new_size_len * 3];
	float* b11 = &memory[new_size_len * 4];
	float* b12 = &memory[new_size_len * 5];
	float* b21 = &memory[new_size_len * 6];
	float* b22 = &memory[new_size_len * 7];

	float* s1 = &memory[new_size_len * 8];
	float* s2 = &memory[new_size_len * 9];
	float* s3 = &memory[new_size_len * 10];
	float* s4 = &memory[new_size_len * 11];
	float* s5 = &memory[new_size_len * 12];
	float* s6 = &memory[new_size_len * 13];
	float* s7 = &memory[new_size_len * 14];
	float* s8 = &memory[new_size_len * 15];
	float* s9 = &memory[new_size_len * 16];
	float* s10 = &memory[new_size_len * 17];

	float* p1 = &memory[new_size_len * 18];
	float* p2 = &memory[new_size_len * 19];
	float* p3 = &memory[new_size_len * 20];
	float* p4 = &memory[new_size_len * 21];
	float* p5 = &memory[new_size_len * 22];
	float* p6 = &memory[new_size_len * 23];
	float* p7 = &memory[new_size_len * 24];

	float* c11 = &memory[new_size_len * 25];
	float* c12 = &memory[new_size_len * 26];
	float* c21 = &memory[new_size_len * 27];
	float* c22 = &memory[new_size_len * 28];

	for (int i = 0; i < new_size; i++) {
		int in = i * n;
		int inn = (i + new_size) * n;
		for (int j = 0; j < new_size; j++) {
			int i2 = i * new_size + j;
			a11[i2] = A[in + j];
			a12[i2] = A[in + j + new_size];
			a21[i2] = A[inn + j];
			a22[i2] = A[inn + j + new_size];

			b11[i2] = B[in + j];
			b12[i2] = B[in + j + new_size];
			b21[i2] = B[inn + j];
			b22[i2] = B[inn + j + new_size];
		}
	}

	subMatf(b12, b22, s1, new_size_len);
	addMatf(a11, a12, s2, new_size_len);
	addMatf(a21, a22, s3, new_size_len);
	subMatf(b21, b11, s4, new_size_len);
	addMatf(a11, a22, s5, new_size_len);
	addMatf(b11, b22, s6, new_size_len);
	subMatf(a12, a22, s7, new_size_len);
	addMatf(b21, b22, s8, new_size_len);
	subMatf(a11, a21, s9, new_size_len);
	addMatf(b11, b12, s10, new_size_len);

	mulMatStrassenRecursive(a11, s1, p1, new_size);
	mulMatStrassenRecursive(s2, b22, p2, new_size);
	mulMatStrassenRecursive(s3, b11, p3, new_size);
	mulMatStrassenRecursive(a22, s4, p4, new_size);
	mulMatStrassenRecursive(s5, s6, p5, new_size);
	mulMatStrassenRecursive(s7, s8, p6, new_size);
	mulMatStrassenRecursive(s9, s10, p7, new_size);

	for (int i = 0; i < new_size_len; i++) {
		c11[i] = p5[i] + p4[i] - p2[i] + p6[i];
		c12[i] = p1[i] + p2[i];
		c21[i] = p3[i] + p4[i];
		c22[i] = p5[i] + p1[i] - p3[i] - p7[i];
	}

	for (int i = 0; i < new_size; i++) {
		int in = i * n;
		int inn = (i + new_size) * n;
		for (int j = 0; j < new_size; j++) {
			C[in + j] = c11[i * new_size + j];
			C[in + j + new_size] = c12[i * new_size + j];
			C[inn + j] = c21[i * new_size + j];
			C[inn + j + new_size] = c22[i * new_size + j];
		}
	}

	delete[] memory;
}

// Multiply each element of the matrix by a scalar
void mulScalarMatf(const float* mat, float scalar, float* out, int n) {
	for (int i = 0; i < n; ++i) {
		out[i] = mat[i] * scalar;
	}
}

// Multiply two matrices element-wise (Hadamard product)
void mulElemwiseMatf(const float* m1, const float* m2, float* out, int n) {
	for (int i = 0; i < n; ++i) {
		out[i] = m1[i] * m2[i];
	}
}

// Divide two matrices element-wise
void divElemwiseMatf(const float* m1, const float* m2, float* out, int n) {
	for (int i = 0; i < n; ++i) {
		out[i] = m1[i] / m2[i];
	}
}
