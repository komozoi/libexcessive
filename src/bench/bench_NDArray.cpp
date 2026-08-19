/*
 * Copyright 2021-2026 komozoi
 * Original Creation Date: 2026-8-18
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or implied, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Logger.h"
#include "NDArray.h"

#include <chrono>
#include <cstdint>

static volatile int64_t gSink;

static int64_t elapsedNs(std::chrono::steady_clock::time_point t0,
                         std::chrono::steady_clock::time_point t1) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

static void report(LogEndpoint& log, const char* op, const char* size, int reps, int64_t totalNs) {
	int64_t per = totalNs / (int64_t)reps;
	if (per >= 1000000)
		log.info("%s  %s  reps=%d  %lld ns/op  (%lld us)", op, size, reps,
		         (long long)per, (long long)(per / 1000));
	else
		log.info("%s  %s  reps=%d  %lld ns/op", op, size, reps, (long long)per);
}

static void fillInt3(NDArray& a, int start) {
	const size_t n = a.numElements();
	for (size_t i = 0; i < n; ++i)
		a.setFlat((size_t)i, (int)((start + (int)i) % 7) - 3);
}

static void fillBinary(NDArray& a, int start) {
	const size_t n = a.numElements();
	for (size_t i = 0; i < n; ++i)
		a.setFlat((size_t)i, (int)((start + (int)i) & 1));
}

static void fillF32(NDArray& a, int start) {
	const size_t n = a.numElements();
	for (size_t i = 0; i < n; ++i)
		a.setFlat((size_t)i, (float)((start + (int)i) % 17) * 0.1f);
}

static void benchInt3Dot(LogEndpoint& log) {
	const int n = 65536;
	const int warmup = 4;
	const int reps = 40;
	NDArray a({n}, INT3);
	NDArray b({n}, INT3);
	fillInt3(a, 0);
	fillInt3(b, 3);
	for (int i = 0; i < warmup; ++i)
		gSink += (int64_t)a.dot<int32_t>(b);
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i)
		gSink += (int64_t)a.dot<int32_t>(b);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "INT3 dot", "n=65536", reps, elapsedNs(t0, t1));
}

static void benchF32Gemm(LogEndpoint& log) {
	const int n = 192;
	const int warmup = 2;
	const int reps = 8;
	NDArray a({n, n}, F32);
	NDArray b({n, n}, F32);
	fillF32(a, 1);
	fillF32(b, 4);
	for (int i = 0; i < warmup; ++i) {
		NDArray c = a.matmul(b);
		gSink += (int64_t)c.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		NDArray c = a.matmul(b);
		gSink += (int64_t)c.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "F32 GEMM", "192x192x192", reps, elapsedNs(t0, t1));
}

static void benchF32Gemv(LogEndpoint& log) {
	const int m = 1024;
	const int k = 1024;
	const int warmup = 4;
	const int reps = 20;
	NDArray a({m, k}, F32);
	NDArray x({k}, F32);
	fillF32(a, 2);
	fillF32(x, 5);
	for (int i = 0; i < warmup; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "F32 GEMV", "1024x1024 @ 1024", reps, elapsedNs(t0, t1));
}

static void benchF32Gemv256(LogEndpoint& log) {
	const int n = 256;
	const int warmup = 8;
	const int reps = 40;
	NDArray a({n, n}, F32);
	NDArray x({n}, F32);
	fillF32(a, 2);
	fillF32(x, 5);
	for (int i = 0; i < warmup; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "F32 GEMV", "256x256 @ 256", reps, elapsedNs(t0, t1));
}

static void benchF64Gemv256(LogEndpoint& log) {
	const int n = 256;
	const int warmup = 8;
	const int reps = 40;
	NDArray a({n, n}, F64);
	NDArray x({n}, F64);
	const size_t nElem = a.numElements();
	for (size_t i = 0; i < nElem; ++i)
		a.setFlat(i, (double)((int)i % 17) * 0.1);
	for (int i = 0; i < n; ++i)
		x.setFlat((size_t)i, (double)((i + 3) % 11) * 0.1);
	for (int i = 0; i < warmup; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<double>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<double>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "F64 GEMV", "256x256 @ 256", reps, elapsedNs(t0, t1));
}

static void benchInt3Gemv(LogEndpoint& log) {
	const int n = 256;
	const int warmup = 4;
	const int reps = 20;
	NDArray a({n, n}, INT3);
	NDArray x({n}, INT3);
	fillInt3(a, 1);
	fillInt3(x, 2);
	for (int i = 0; i < warmup; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<int>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		NDArray y = a.gemv(x);
		gSink += (int64_t)y.getFlat<int>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "INT3 GEMV", "256x256 @ 256", reps, elapsedNs(t0, t1));
}

static void benchBinaryHamming(LogEndpoint& log) {
	const int n = 65536;
	const int warmup = 4;
	const int reps = 40;
	NDArray a({n}, BINARY);
	NDArray b({n}, BINARY);
	fillBinary(a, 0);
	fillBinary(b, 1);
	for (int i = 0; i < warmup; ++i)
		gSink += (int64_t)a.hamming<int32_t>(b);
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i)
		gSink += (int64_t)a.hamming<int32_t>(b);
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "BINARY hamming", "n=65536", reps, elapsedNs(t0, t1));
}

static void restoreF32(NDArray& dst, const NDArray& src) {
	const size_t n = src.numElements();
	const float* s = src.data<float>();
	float* d = dst.data<float>();
	for (size_t i = 0; i < n; ++i)
		d[i] = s[i];
}

static void benchF32Softmax(LogEndpoint& log) {
	const int rows = 2048;
	const int cols = 256;
	const int warmup = 2;
	const int reps = 8;
	NDArray src({rows, cols}, F32);
	NDArray a({rows, cols}, F32);
	fillF32(src, 1);
	restoreF32(a, src);
	for (int i = 0; i < warmup; ++i) {
		restoreF32(a, src);
		a.softmax();
		gSink += (int64_t)a.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		restoreF32(a, src);
		a.softmax();
		gSink += (int64_t)a.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "F32 softmax", "2048x256", reps, elapsedNs(t0, t1));
}

static void benchF32Rmsnorm(LogEndpoint& log) {
	const int rows = 2048;
	const int cols = 256;
	const int warmup = 2;
	const int reps = 8;
	NDArray x({rows, cols}, F32);
	NDArray w({cols}, F32);
	fillF32(x, 2);
	fillF32(w, 3);
	for (int i = 0; i < warmup; ++i) {
		NDArray y = x.rmsnorm(w, 1e-6f);
		gSink += (int64_t)y.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		NDArray y = x.rmsnorm(w, 1e-6f);
		gSink += (int64_t)y.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "F32 rmsnorm", "2048x256", reps, elapsedNs(t0, t1));
}

static void benchF32Silu(LogEndpoint& log) {
	const int rows = 2048;
	const int cols = 256;
	const int warmup = 2;
	const int reps = 8;
	NDArray src({rows, cols}, F32);
	NDArray a({rows, cols}, F32);
	fillF32(src, 4);
	restoreF32(a, src);
	for (int i = 0; i < warmup; ++i) {
		restoreF32(a, src);
		a.silu();
		gSink += (int64_t)a.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < reps; ++i) {
		restoreF32(a, src);
		a.silu();
		gSink += (int64_t)a.getFlat<float>(0);
	}
	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	report(log, "F32 silu", "2048x256", reps, elapsedNs(t0, t1));
}

int main() {
	Logger logger("logs", LOG_LEVEL_CRIT, LOG_LEVEL_INFO);
	LogEndpoint log(logger, "ndarray-bench");
	benchInt3Dot(log);
	benchF32Gemm(log);
	benchF32Gemv(log);
	benchF32Gemv256(log);
	benchF64Gemv256(log);
	benchInt3Gemv(log);
	benchBinaryHamming(log);
	benchF32Softmax(log);
	benchF32Rmsnorm(log);
	benchF32Silu(log);
	return 0;
}
