// Copyright 2021-2026 komozoi
// Softmax / RMSNorm / SiLU

#include <cmath>
#include <gtest/gtest.h>

#include "NDArray.h"


static float siluRef(float x) {
	return x / (1.0f + std::exp(-x));
}

TEST(NDArray_activations, Softmax_RowSumsToOne) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	a.softmax();
	EXPECT_EQ(a.type, F32);
	float s = a.get<float>({0}) + a.get<float>({1}) + a.get<float>({2});
	EXPECT_NEAR(s, 1.0f, 1e-6f);
	EXPECT_GT(a.get<float>({2}), a.get<float>({1}));
	EXPECT_GT(a.get<float>({1}), a.get<float>({0}));
	const float e0 = std::exp(1.0f - 3.0f);
	const float e1 = std::exp(2.0f - 3.0f);
	const float e2 = 1.0f;
	const float z = e0 + e1 + e2;
	EXPECT_NEAR(a.get<float>({0}), e0 / z, 1e-6f);
	EXPECT_NEAR(a.get<float>({1}), e1 / z, 1e-6f);
	EXPECT_NEAR(a.get<float>({2}), e2 / z, 1e-6f);
}

TEST(NDArray_activations, Softmax_StableLargeLogits) {
	NDArray a(ArrayList({1000.0f, 1001.0f, 1002.0f}));
	a.softmax();
	float s = a.get<float>({0}) + a.get<float>({1}) + a.get<float>({2});
	EXPECT_NEAR(s, 1.0f, 1e-5f);
	EXPECT_TRUE(std::isfinite(a.get<float>({0})));
	EXPECT_TRUE(std::isfinite(a.get<float>({2})));
}

TEST(NDArray_activations, Softmax_LastAxisMatrix) {
	NDArray a({2, 3}, ArrayList({
		1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f
	}));
	a.softmax();
	EXPECT_NEAR(a.get<float>({0, 0}) + a.get<float>({0, 1}) + a.get<float>({0, 2}), 1.0f, 1e-6f);
	EXPECT_NEAR(a.get<float>({1, 0}) + a.get<float>({1, 1}) + a.get<float>({1, 2}), 1.0f, 1e-6f);
	EXPECT_GT(a.get<float>({0, 0}), a.get<float>({0, 1}));
	EXPECT_GT(a.get<float>({1, 2}), a.get<float>({1, 0}));
}

TEST(NDArray_activations, Softmaxed_DoesNotMutate) {
	NDArray a(ArrayList({0.0f, 1.0f}));
	NDArray b = a.softmaxed();
	EXPECT_FLOAT_EQ(a.get<float>({0}), 0.0f);
	EXPECT_NEAR(b.get<float>({0}) + b.get<float>({1}), 1.0f, 1e-6f);
}

TEST(NDArray_activations, Softmax_ScalarIsOne) {
	NDArray s(F32, 7.0f);
	s.softmax();
	EXPECT_FLOAT_EQ(s.get<float>({}), 1.0f);
}

TEST(NDArray_activations, RMSNorm_KnownVector) {
	NDArray x(ArrayList({3.0f, 4.0f}));
	NDArray w(ArrayList({1.0f, 1.0f}));
	NDArray y = x.rmsnorm(w, 0.0f);
	const float rms = std::sqrt((9.0f + 16.0f) / 2.0f);
	EXPECT_EQ(y.type, F32);
	EXPECT_NEAR(y.get<float>({0}), 3.0f / rms, 1e-6f);
	EXPECT_NEAR(y.get<float>({1}), 4.0f / rms, 1e-6f);
	EXPECT_FLOAT_EQ(x.get<float>({0}), 3.0f);
}

TEST(NDArray_activations, RMSNorm_Weighted) {
	NDArray x(ArrayList({2.0f, 2.0f}));
	NDArray w(ArrayList({1.0f, 3.0f}));
	NDArray y = x.rmsnorm(w, 0.0f);
	const float rms = std::sqrt((4.0f + 4.0f) / 2.0f);
	EXPECT_NEAR(y.get<float>({0}), 2.0f * 1.0f / rms, 1e-6f);
	EXPECT_NEAR(y.get<float>({1}), 2.0f * 3.0f / rms, 1e-6f);
}

TEST(NDArray_activations, SiLU_MatchesFormula) {
	NDArray a(ArrayList({0.0f, 1.0f, -2.0f, 4.0f}));
	a.silu();
	EXPECT_NEAR(a.get<float>({0}), siluRef(0.0f), 1e-6f);
	EXPECT_NEAR(a.get<float>({1}), siluRef(1.0f), 1e-6f);
	EXPECT_NEAR(a.get<float>({2}), siluRef(-2.0f), 1e-6f);
	EXPECT_NEAR(a.get<float>({3}), siluRef(4.0f), 1e-5f);
}

TEST(NDArray_activations, Silued_DoesNotMutate) {
	NDArray a(ArrayList({1.0f}));
	NDArray b = a.silued();
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
	EXPECT_NEAR(b.get<float>({0}), siluRef(1.0f), 1e-6f);
}

TEST(NDArray_activations, SiluMul) {
	NDArray a(ArrayList({1.0f, -1.0f, 2.0f}));
	NDArray g(ArrayList({0.5f, 2.0f, 1.0f}));
	a.siluMul(g);
	EXPECT_NEAR(a.get<float>({0}), siluRef(1.0f) * 0.5f, 1e-6f);
	EXPECT_NEAR(a.get<float>({1}), siluRef(-1.0f) * 2.0f, 1e-6f);
	EXPECT_NEAR(a.get<float>({2}), siluRef(2.0f) * 1.0f, 1e-6f);
}

TEST(NDArray_activations, Softmax_F16_StaysF16) {
	NDArray a({3}, F16);
	a.set({0}, 1.0f);
	a.set({1}, 2.0f);
	a.set({2}, 3.0f);
	a.softmax();
	EXPECT_EQ(a.type, F16);
	float s = a.get<float>({0}) + a.get<float>({1}) + a.get<float>({2});
	EXPECT_NEAR(s, 1.0f, 2e-3f);
}

static void softmaxRefF32(float* p, int n) {
	float m = p[0];
	for (int i = 1; i < n; ++i)
		if (p[i] > m)
			m = p[i];
	float z = 0.0f;
	for (int i = 0; i < n; ++i) {
		p[i] = std::exp(p[i] - m);
		z += p[i];
	}
	const float inv = z == 0.0f ? 0.0f : 1.0f / z;
	for (int i = 0; i < n; ++i)
		p[i] *= inv;
}

TEST(NDArray_activations, Softmax_F32_LongRow_MatchesRef) {
	const int n = 256;
	NDArray a({n}, F32);
	float ref[256];
	for (int i = 0; i < n; ++i) {
		float v = 0.05f * (float)(i - 80);
		a.set({i}, v);
		ref[i] = v;
	}
	softmaxRefF32(ref, n);
	a.softmax();
	float sum = 0.0f;
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(a.get<float>({i}), ref[i], 2e-5f);
		sum += a.get<float>({i});
	}
	EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(NDArray_activations, Softmax_INT8_PromotesToF32) {
	NDArray a({4}, INT8);
	a.set({0}, 0);
	a.set({1}, 1);
	a.set({2}, 2);
	a.set({3}, 1);
	a.softmax();
	EXPECT_EQ(a.type, F32);
	float s = 0.0f;
	for (int i = 0; i < 4; ++i)
		s += a.get<float>({i});
	EXPECT_NEAR(s, 1.0f, 1e-5f);
}

TEST(NDArray_activations, RMSNorm_F32_Rows_MatchRef) {
	NDArray x({2, 4}, ArrayList({
		1.0f, 2.0f, 3.0f, 4.0f,
		0.5f, -0.5f, 1.5f, -1.5f
	}));
	NDArray w(ArrayList({1.0f, 2.0f, 0.5f, 1.0f}));
	NDArray y = x.rmsnorm(w, 1e-6f);
	EXPECT_EQ(y.type, F32);
	for (int r = 0; r < 2; ++r) {
		float ss = 0.0f;
		for (int c = 0; c < 4; ++c) {
			float v = x.get<float>({r, c});
			ss += v * v;
		}
		const float inv = 1.0f / std::sqrt(ss / 4.0f + 1e-6f);
		for (int c = 0; c < 4; ++c)
			EXPECT_NEAR(y.get<float>({r, c}),
			            x.get<float>({r, c}) * w.get<float>({c}) * inv, 1e-5f);
	}
}

TEST(NDArray_activations, Softmax_F16_Matrix_LastAxis) {
	NDArray a({2, 4}, F16);
	const float v[8] = {1.0f, 2.0f, 0.0f, 3.0f, 0.0f, 0.0f, 4.0f, 1.0f};
	for (int i = 0; i < 8; ++i)
		a.setFlat((size_t)i, v[i]);
	a.softmax();
	EXPECT_EQ(a.type, F16);
	EXPECT_NEAR(a.get<float>({0, 0}) + a.get<float>({0, 1})
	            + a.get<float>({0, 2}) + a.get<float>({0, 3}), 1.0f, 3e-3f);
	EXPECT_NEAR(a.get<float>({1, 0}) + a.get<float>({1, 1})
	            + a.get<float>({1, 2}) + a.get<float>({1, 3}), 1.0f, 3e-3f);
}

TEST(NDArray_activations, SiLU_F16_StaysF16) {
	NDArray a({3}, F16);
	a.set({0}, 0.0f);
	a.set({1}, 1.0f);
	a.set({2}, -1.0f);
	a.silu();
	EXPECT_EQ(a.type, F16);
	EXPECT_NEAR(a.get<float>({0}), siluRef(0.0f), 2e-3f);
	EXPECT_NEAR(a.get<float>({1}), siluRef(1.0f), 2e-3f);
	EXPECT_NEAR(a.get<float>({2}), siluRef(-1.0f), 2e-3f);
}
