// INTERNAL — include only from NDArray.cpp / ndarray_matmul.cpp.

#ifndef LIBEXCESSIVE_SRC_NDARRAY_MATMUL_H
#define LIBEXCESSIVE_SRC_NDARRAY_MATMUL_H

class NDArray;
class NDArrayView;

NDArray ndmatmul(const NDArrayView& a, const NDArrayView& b);

#endif
