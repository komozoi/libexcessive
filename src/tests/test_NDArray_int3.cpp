// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-4
//
// All rights reserved.
//
// INT3 is exercised only through the public NDArray API (no private int3 headers).

#include <gtest/gtest.h>
#include <stdexcept>

#include "NDArray.h"

/** Wrap to INT3 range via NDArray (encode/decode of low 3 bits). */
static int wrap3(int v) {
	return (int)(int8_t)((v & 7) << 5) >> 5; // sign-extend 3-bit field
}

TEST(NDArray_int3, ConstructSetGet) {
	NDArray a({8}, INT3);
	for (int i = 0; i < 8; ++i)
		a.set({i}, i - 4);
	for (int i = 0; i < 8; ++i)
		EXPECT_EQ(a.get<int>({i}), i - 4);
	EXPECT_EQ(a.getFlat<int>(0), -4);
	EXPECT_EQ(a.getFlat<int>(7), 3);
}

TEST(NDArray_int3, AddSubMulDiv_Array) {
	NDArray a({4}, INT3);
	NDArray b({4}, INT3);
	a.set({0}, 1); a.set({1}, 2); a.set({2}, -3); a.set({3}, -4);
	b.set({0}, 2); b.set({1}, -1); b.set({2}, 2); b.set({3}, 1);

	NDArray sum = a + b;
	EXPECT_EQ(sum.type, INT3);
	EXPECT_EQ(sum.get<int>({0}), 3);
	EXPECT_EQ(sum.get<int>({1}), 1);
	EXPECT_EQ(sum.get<int>({2}), -1);
	EXPECT_EQ(sum.get<int>({3}), -3);

	NDArray diff = a - b;
	EXPECT_EQ(diff.get<int>({0}), -1);
	EXPECT_EQ(diff.get<int>({1}), 3);
	EXPECT_EQ(diff.get<int>({2}), wrap3(-3 - 2)); // -5 → 3
	EXPECT_EQ(diff.get<int>({3}), wrap3(-4 - 1));

	NDArray prod = a * b;
	EXPECT_EQ(prod.get<int>({0}), 2);
	EXPECT_EQ(prod.get<int>({1}), -2);
	EXPECT_EQ(prod.get<int>({2}), wrap3((-3) * 2)); // -6 → 2
	EXPECT_EQ(prod.get<int>({3}), -4);

	NDArray quot = a / b;
	EXPECT_EQ(quot.get<int>({0}), 0);
	EXPECT_EQ(quot.get<int>({1}), -2);
	EXPECT_EQ(quot.get<int>({2}), -1);
	EXPECT_EQ(quot.get<int>({3}), -4);
}

TEST(NDArray_int3, DivByZero_Throws) {
	NDArray a({2}, INT3);
	NDArray b({2}, INT3);
	a.set({0}, 1); a.set({1}, 2);
	b.set({0}, 1); b.set({1}, 0);
	EXPECT_THROW(a / b, std::invalid_argument);
}

TEST(NDArray_int3, ScalarOps) {
	NDArray a({3}, INT3);
	a.set({0}, 1); a.set({1}, -2); a.set({2}, 3);
	// Int scalar promotes INT3 → INT32 (lossless policy).
	NDArray r = a + 1;
	EXPECT_EQ(r.type, INT32);
	EXPECT_EQ(r.get<int>({0}), 2);
	EXPECT_EQ(r.get<int>({1}), -1);
	EXPECT_EQ(r.get<int>({2}), 4);

	// Same-type wrap stays INT3.
	NDArray w({3}, INT3);
	w.set({0}, 1); w.set({1}, -2); w.set({2}, 3);
	w.add(NDArray::full(ArrayList<int>({3}), INT3, 1));
	EXPECT_EQ(w.type, INT3);
	EXPECT_EQ(w.get<int>({0}), 2);
	EXPECT_EQ(w.get<int>({1}), -1);
	EXPECT_EQ(w.get<int>({2}), wrap3(3 + 1));

	NDArray m = a * NDArray::full(ArrayList<int>({3}), INT3, 2);
	EXPECT_EQ(m.type, INT3);
	EXPECT_EQ(m.get<int>({0}), 2);
	EXPECT_EQ(m.get<int>({1}), -4);
	EXPECT_EQ(m.get<int>({2}), wrap3(3 * 2));
}

TEST(NDArray_int3, PromoteWithFloat) {
	NDArray a({2}, INT3);
	a.set({0}, -4); a.set({1}, 3);
	NDArray b(ArrayList<float>({10.0f, 20.0f}));
	NDArray r = a + b;
	EXPECT_EQ(r.type, F32);
	EXPECT_FLOAT_EQ(r.get<float>({0}), 6.0f);
	EXPECT_FLOAT_EQ(r.get<float>({1}), 23.0f);
}

TEST(NDArray_int3, ConvertRoundTrip) {
	NDArray a({5}, INT3);
	for (int i = 0; i < 5; ++i)
		a.set({i}, i - 2);
	NDArray wide = a.convert(INT32);
	EXPECT_EQ(wide.type, INT32);
	for (int i = 0; i < 5; ++i)
		EXPECT_EQ(wide.get<int>({i}), i - 2);
	NDArray back = wide.convert(INT3);
	for (int i = 0; i < 5; ++i)
		EXPECT_EQ(back.get<int>({i}), i - 2);
}

TEST(NDArray_int3, LongVector_MatchesScalarSemantics) {
	const int N = 200;
	NDArray a({N}, INT3);
	NDArray b({N}, INT3);
	for (int i = 0; i < N; ++i) {
		a.set({i}, (i % 7) - 3);
		int bv = ((i * 3) % 7) - 3;
		if (bv == 0)
			bv = 1;
		b.set({i}, bv);
	}
	NDArray sum = a + b;
	NDArray dif = a - b;
	NDArray prod = a * b;
	NDArray quot = a / b;
	for (int i = 0; i < N; ++i) {
		int av = a.get<int>({i});
		int bv = b.get<int>({i});
		EXPECT_EQ(sum.get<int>({i}), wrap3(av + bv));
		EXPECT_EQ(dif.get<int>({i}), wrap3(av - bv));
		EXPECT_EQ(prod.get<int>({i}), wrap3(av * bv));
		EXPECT_EQ(quot.get<int>({i}), av / bv);
	}
}

TEST(NDArray_int3, NegAbs) {
	NDArray a({4}, INT3);
	a.set({0}, 1); a.set({1}, -2); a.set({2}, 3); a.set({3}, -4);
	NDArray n = a.negated();
	EXPECT_EQ(n.get<int>({0}), -1);
	EXPECT_EQ(n.get<int>({1}), 2);
	EXPECT_EQ(n.get<int>({2}), -3);
	EXPECT_EQ(n.get<int>({3}), -4); // -(-4) wraps to -4 in INT3
	NDArray ab = a.absolute();
	EXPECT_EQ(ab.get<int>({0}), 1);
	EXPECT_EQ(ab.get<int>({1}), 2);
	EXPECT_EQ(ab.get<int>({2}), 3);
	EXPECT_EQ(ab.get<int>({3}), -4); // abs(-4) wraps
}

TEST(NDArray_int3, DefaultFactories_UseInt3) {
	NDArray z = NDArray::zeros({4});
	EXPECT_EQ(z.type, INT3);
	EXPECT_EQ(z.get<int>({0}), 0);
	EXPECT_EQ(z.get<int>({3}), 0);

	NDArray o = NDArray::ones({3});
	EXPECT_EQ(o.type, INT3);
	EXPECT_EQ(o.get<int>({0}), 1);
	EXPECT_EQ(o.get<int>({2}), 1);

	NDArray m1 = NDArray::full({2}, -1);
	EXPECT_EQ(m1.type, INT3);
	EXPECT_EQ(m1.get<int>({0}), -1);

	NDArray big = NDArray::full({2}, 42);
	EXPECT_EQ(big.type, INT32);
	EXPECT_EQ(big.get<int>({0}), 42);

	// Explicit type still works
	NDArray zf = NDArray::zeros({2}, F32);
	EXPECT_EQ(zf.type, F32);
}

TEST(NDArray_int3, ScalarAndVectorCtors_Ternary) {
	NDArray s0(0);
	NDArray s1(1);
	NDArray sm(-1);
	EXPECT_EQ(s0.type, INT3);
	EXPECT_EQ(s1.type, INT3);
	EXPECT_EQ(sm.type, INT3);
	EXPECT_EQ(s0.get<int>({}), 0);
	EXPECT_EQ(s1.get<int>({}), 1);
	EXPECT_EQ(sm.get<int>({}), -1);

	NDArray other(7);
	EXPECT_EQ(other.type, INT32);
	EXPECT_EQ(other.get<int>({}), 7);

	NDArray v(ArrayList<int32_t>({-1, 0, 1, 1, 0, -1}));
	EXPECT_EQ(v.type, INT3);
	EXPECT_EQ(v.get<int>({0}), -1);
	EXPECT_EQ(v.get<int>({2}), 1);

	NDArray w(ArrayList<int32_t>({0, 2, 1}));
	EXPECT_EQ(w.type, INT32);
}

TEST(NDArray_int3, Compare_ArrayAndScalars) {
	NDArray a(ArrayList<float>({1.0f, 5.0f, 3.0f}));
	NDArray b(ArrayList<float>({2.0f, 5.0f, 1.0f}));
	NDArray c = a.compare(b);
	EXPECT_EQ(c.type, INT3);
	EXPECT_EQ(c.get<int>({0}), -1); // 1 < 2
	EXPECT_EQ(c.get<int>({1}), 0);  // equal
	EXPECT_EQ(c.get<int>({2}), 1);  // 3 > 1

	NDArray vs = a.compare(3.0f);
	EXPECT_EQ(vs.type, INT3);
	EXPECT_EQ(vs.get<int>({0}), -1);
	EXPECT_EQ(vs.get<int>({1}), 1);
	EXPECT_EQ(vs.get<int>({2}), 0);

	NDArray vi = a.compare(5);
	EXPECT_EQ(vi.get<int>({0}), -1);
	EXPECT_EQ(vi.get<int>({1}), 0);
	EXPECT_EQ(vi.get<int>({2}), -1);

	NDArray i3a({3}, INT3);
	i3a.set({0}, -1); i3a.set({1}, 0); i3a.set({2}, 1);
	NDArray i3b = NDArray::ones({3}); // INT3 ones
	NDArray ic = i3a.compare(i3b);
	EXPECT_EQ(ic.get<int>({0}), -1);
	EXPECT_EQ(ic.get<int>({1}), -1);
	EXPECT_EQ(ic.get<int>({2}), 0);

	NDArray u({2}, UINT256);
	u.set({0}, uint256_t(10));
	u.set({1}, uint256_t(3));
	NDArray uc = u.compare(uint256_t(5));
	EXPECT_EQ(uc.get<int>({0}), 1);
	EXPECT_EQ(uc.get<int>({1}), -1);
}

TEST(NDArray_int3, MinMax_MatchScalarReference) {
	NDArray a = NDArray::full({5}, INT3, 0);
	const int vals[] = {1, -4, 3, -2, 0};
	for (int i = 0; i < 5; ++i)
		a.set({i}, vals[i]);

	int expectMin = vals[0];
	int expectMax = vals[0];
	for (int i = 1; i < 5; ++i) {
		if (vals[i] < expectMin)
			expectMin = vals[i];
		if (vals[i] > expectMax)
			expectMax = vals[i];
	}

	NDArray mn = a.min();
	NDArray mx = a.max();
	EXPECT_EQ(mn.type, INT3);
	EXPECT_EQ(mx.type, INT3);
	EXPECT_EQ(mn.shape.size(), 0);
	EXPECT_EQ(mn.get<int>({}), expectMin);
	EXPECT_EQ(mx.get<int>({}), expectMax);
}

TEST(NDArray_int3, MinMax_Axis) {
	NDArray m({2, 3}, INT3);
	const int vals[2][3] = {
		{-4, 2, 1},
		{3, -1, 0}
	};
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			m.set({i, j}, vals[i][j]);

	NDArray mn0 = m.min(0);
	EXPECT_EQ(mn0.type, INT3);
	EXPECT_EQ(mn0.get<int>({0}), -4);
	EXPECT_EQ(mn0.get<int>({1}), -1);
	EXPECT_EQ(mn0.get<int>({2}), 0);

	NDArray mx1 = m.max(1);
	EXPECT_EQ(mx1.get<int>({0}), 2);
	EXPECT_EQ(mx1.get<int>({1}), 3);
}

TEST(NDArray_int3, Sign) {
	NDArray a({5}, INT3);
	const int vals[] = {-4, -1, 0, 1, 3};
	for (int i = 0; i < 5; ++i)
		a.set({i}, vals[i]);
	a.sign();
	EXPECT_EQ(a.type, INT3);
	EXPECT_EQ(a.get<int>({0}), -1);
	EXPECT_EQ(a.get<int>({1}), -1);
	EXPECT_EQ(a.get<int>({2}), 0);
	EXPECT_EQ(a.get<int>({3}), 1);
	EXPECT_EQ(a.get<int>({4}), 1);
}

TEST(NDArray_int3, BroadcastAdd_Size1) {
	NDArray a({2, 3}, INT3);
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			a.set({i, j}, 1);
	NDArray b({2, 1}, INT3);
	b.set({0, 0}, 2);
	b.set({1, 0}, -3);

	a.broadcastAdd(b);
	EXPECT_EQ(a.type, INT3);
	EXPECT_EQ(a.get<int>({0, 0}), 3);
	EXPECT_EQ(a.get<int>({0, 1}), 3);
	EXPECT_EQ(a.get<int>({0, 2}), 3);
	EXPECT_EQ(a.get<int>({1, 0}), wrap3(1 + -3));
	EXPECT_EQ(a.get<int>({1, 1}), wrap3(1 + -3));
	EXPECT_EQ(a.get<int>({1, 2}), wrap3(1 + -3));
}

TEST(NDArray_int3, BroadcastAdd_Trailing) {
	NDArray a({2, 3}, INT3);
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			a.set({i, j}, 1);
	NDArray b({3}, INT3);
	b.set({0}, 2);
	b.set({1}, -3);
	b.set({2}, 1);

	a.broadcastAdd(b);
	EXPECT_EQ(a.get<int>({0, 0}), 3);
	EXPECT_EQ(a.get<int>({0, 1}), wrap3(1 + -3));
	EXPECT_EQ(a.get<int>({0, 2}), 2);
	EXPECT_EQ(a.get<int>({1, 0}), 3);
	EXPECT_EQ(a.get<int>({1, 1}), wrap3(1 + -3));
	EXPECT_EQ(a.get<int>({1, 2}), 2);
}

TEST(NDArray_int3, BroadcastAdd_WrapsLikeElemwise) {
	NDArray a({2, 2}, INT3);
	a.set({0, 0}, 3);
	a.set({0, 1}, 3);
	a.set({1, 0}, -4);
	a.set({1, 1}, -4);
	NDArray b({2, 1}, INT3);
	b.set({0, 0}, 2);
	b.set({1, 0}, -1);

	a.broadcastAdd(b);
	EXPECT_EQ(a.get<int>({0, 0}), wrap3(3 + 2));
	EXPECT_EQ(a.get<int>({0, 1}), wrap3(3 + 2));
	EXPECT_EQ(a.get<int>({1, 0}), wrap3(-4 + -1));
	EXPECT_EQ(a.get<int>({1, 1}), wrap3(-4 + -1));
}

TEST(NDArray_int3, BroadcastDiv_Zero_Throws) {
	NDArray a({2, 2}, INT3);
	a.set({0, 0}, 1);
	a.set({0, 1}, 1);
	a.set({1, 0}, 2);
	a.set({1, 1}, 2);
	NDArray b({2}, INT3);
	b.set({0}, 1);
	b.set({1}, 0);
	EXPECT_THROW(a.broadcastDiv(b), std::invalid_argument);
}
