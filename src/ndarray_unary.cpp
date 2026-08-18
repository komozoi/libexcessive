// Packed exp / log / sin / sqrt. Float libm in restrict 4-wide loops
// so GCC/Clang can emit libmvec when this TU is built -fno-math-errno.
// F16 / BF16 convert a block to F32, apply, convert back.

#include "ndarray_unary.h"
#include "ndarray_half_kernels.h"

#include <cmath>

#if defined(_MSC_VER)
#define NDU_RESTRICT __restrict
#else
#define NDU_RESTRICT __restrict__
#endif

static const float kDeg2RadF = 3.14159265358979323846f / 180.0f;
static const float kRad2DegF = 180.0f / 3.14159265358979323846f;
static const double kDeg2RadD = 3.14159265358979323846 / 180.0;
static const double kRad2DegD = 180.0 / 3.14159265358979323846;

#define NDU_LOOP(fn) \
	do { \
		for (size_t i = 0; i < n; ++i) \
			p[i] = fn(p[i]); \
	} while (0)

static float ndu_deg2rad_f(float x) { return x * kDeg2RadF; }
static float ndu_rad2deg_f(float x) { return x * kRad2DegF; }
static double ndu_deg2rad_d(double x) { return x * kDeg2RadD; }
static double ndu_rad2deg_d(double x) { return x * kRad2DegD; }

static void ndu_f32_exp(float* NDU_RESTRICT p, size_t n) {
#pragma omp simd
	for (size_t i = 0; i < n; ++i)
		p[i] = std::exp(p[i]);
}
static void ndu_f32_log(float* NDU_RESTRICT p, size_t n) {
#pragma omp simd
	for (size_t i = 0; i < n; ++i)
		p[i] = std::log(p[i]);
}
static void ndu_f32_sin(float* NDU_RESTRICT p, size_t n) {
#pragma omp simd
	for (size_t i = 0; i < n; ++i)
		p[i] = std::sin(p[i]);
}
static void ndu_f32_sqrt(float* NDU_RESTRICT p, size_t n) {
#pragma omp simd
	for (size_t i = 0; i < n; ++i)
		p[i] = std::sqrt(p[i]);
}

void ndunary_f32(float* NDU_RESTRICT p, size_t n, NDUnaryOp op) {
	switch (op) {
		case NDUnaryOp::Sqrt:    ndu_f32_sqrt(p, n); break;
		case NDUnaryOp::Cbrt:    NDU_LOOP(std::cbrt); break;
		case NDUnaryOp::Exp:     ndu_f32_exp(p, n); break;
		case NDUnaryOp::Expm1:   NDU_LOOP(std::expm1); break;
		case NDUnaryOp::Log:     ndu_f32_log(p, n); break;
		case NDUnaryOp::Log2:    NDU_LOOP(std::log2); break;
		case NDUnaryOp::Log10:   NDU_LOOP(std::log10); break;
		case NDUnaryOp::Log1p:   NDU_LOOP(std::log1p); break;
		case NDUnaryOp::Sin:     ndu_f32_sin(p, n); break;
		case NDUnaryOp::Cos:     NDU_LOOP(std::cos); break;
		case NDUnaryOp::Tan:     NDU_LOOP(std::tan); break;
		case NDUnaryOp::Asin:    NDU_LOOP(std::asin); break;
		case NDUnaryOp::Acos:    NDU_LOOP(std::acos); break;
		case NDUnaryOp::Atan:    NDU_LOOP(std::atan); break;
		case NDUnaryOp::Sinh:    NDU_LOOP(std::sinh); break;
		case NDUnaryOp::Cosh:    NDU_LOOP(std::cosh); break;
		case NDUnaryOp::Tanh:    NDU_LOOP(std::tanh); break;
		case NDUnaryOp::Asinh:   NDU_LOOP(std::asinh); break;
		case NDUnaryOp::Acosh:   NDU_LOOP(std::acosh); break;
		case NDUnaryOp::Atanh:   NDU_LOOP(std::atanh); break;
		case NDUnaryOp::Deg2Rad: NDU_LOOP(ndu_deg2rad_f); break;
		case NDUnaryOp::Rad2Deg: NDU_LOOP(ndu_rad2deg_f); break;
		case NDUnaryOp::Floor:   NDU_LOOP(std::floor); break;
		case NDUnaryOp::Ceil:    NDU_LOOP(std::ceil); break;
		case NDUnaryOp::Round:   NDU_LOOP(std::round); break;
	}
}

void ndunary_f64(double* NDU_RESTRICT p, size_t n, NDUnaryOp op) {
	switch (op) {
		case NDUnaryOp::Sqrt:    NDU_LOOP(std::sqrt); break;
		case NDUnaryOp::Cbrt:    NDU_LOOP(std::cbrt); break;
		case NDUnaryOp::Exp:     NDU_LOOP(std::exp); break;
		case NDUnaryOp::Expm1:   NDU_LOOP(std::expm1); break;
		case NDUnaryOp::Log:     NDU_LOOP(std::log); break;
		case NDUnaryOp::Log2:    NDU_LOOP(std::log2); break;
		case NDUnaryOp::Log10:   NDU_LOOP(std::log10); break;
		case NDUnaryOp::Log1p:   NDU_LOOP(std::log1p); break;
		case NDUnaryOp::Sin:     NDU_LOOP(std::sin); break;
		case NDUnaryOp::Cos:     NDU_LOOP(std::cos); break;
		case NDUnaryOp::Tan:     NDU_LOOP(std::tan); break;
		case NDUnaryOp::Asin:    NDU_LOOP(std::asin); break;
		case NDUnaryOp::Acos:    NDU_LOOP(std::acos); break;
		case NDUnaryOp::Atan:    NDU_LOOP(std::atan); break;
		case NDUnaryOp::Sinh:    NDU_LOOP(std::sinh); break;
		case NDUnaryOp::Cosh:    NDU_LOOP(std::cosh); break;
		case NDUnaryOp::Tanh:    NDU_LOOP(std::tanh); break;
		case NDUnaryOp::Asinh:   NDU_LOOP(std::asinh); break;
		case NDUnaryOp::Acosh:   NDU_LOOP(std::acosh); break;
		case NDUnaryOp::Atanh:   NDU_LOOP(std::atanh); break;
		case NDUnaryOp::Deg2Rad: NDU_LOOP(ndu_deg2rad_d); break;
		case NDUnaryOp::Rad2Deg: NDU_LOOP(ndu_rad2deg_d); break;
		case NDUnaryOp::Floor:   NDU_LOOP(std::floor); break;
		case NDUnaryOp::Ceil:    NDU_LOOP(std::ceil); break;
		case NDUnaryOp::Round:   NDU_LOOP(std::round); break;
	}
}

#undef NDU_LOOP

static const size_t kHalfBlock = 32;

static void ndunary_half(uint16_t* p, size_t n, NDUnaryOp op, bool isF16) {
	float buf[kHalfBlock];
	for (size_t i = 0; i < n; ) {
		const size_t m = n - i < kHalfBlock ? n - i : kHalfBlock;
		if (isF16)
			ndhalf_f16_to_f32(buf, p + i, m);
		else
			ndhalf_bf16_to_f32(buf, p + i, m);
		ndunary_f32(buf, m, op);
		if (isF16)
			ndhalf_f32_to_f16(p + i, buf, m);
		else
			ndhalf_f32_to_bf16(p + i, buf, m);
		i += m;
	}
}

void ndunary_f16(uint16_t* p, size_t n, NDUnaryOp op) {
	ndunary_half(p, n, op, true);
}

void ndunary_bf16(uint16_t* p, size_t n, NDUnaryOp op) {
	ndunary_half(p, n, op, false);
}
