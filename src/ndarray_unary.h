// INTERNAL — packed real unaries (exp / log / sin / sqrt / …).
// Include from NDArray.cpp / ndarray_unary.cpp.

#ifndef LIBEXCESSIVE_SRC_NDARRAY_UNARY_H
#define LIBEXCESSIVE_SRC_NDARRAY_UNARY_H

#include <cstddef>
#include <cstdint>

enum class NDUnaryOp {
	Sqrt, Cbrt, Exp, Expm1, Log, Log2, Log10, Log1p,
	Sin, Cos, Tan, Asin, Acos, Atan,
	Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
	Deg2Rad, Rad2Deg, Floor, Ceil, Round
};

void ndunary_f32(float* p, size_t n, NDUnaryOp op);
void ndunary_f64(double* p, size_t n, NDUnaryOp op);
void ndunary_f16(uint16_t* p, size_t n, NDUnaryOp op);
void ndunary_bf16(uint16_t* p, size_t n, NDUnaryOp op);

#endif
