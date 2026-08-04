
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.

#include <gtest/gtest.h>

#include "NDArray.h"


TEST(NDArray_indexing, ArrayList_GetSet) {
	NDArray m({2, 3}, F32);
	ArrayList<int> ij;
	ij.add(1);
	ij.add(2);
	m.set(ij, 42.0f);
	EXPECT_FLOAT_EQ(m.get<float>(ij), 42.0f);
	EXPECT_FLOAT_EQ(m.get<float>({1, 2}), 42.0f);
}

TEST(NDArray_indexing, Flat_GetSet) {
	NDArray a({2, 3}, F32);
	// row-major: (1,0) → flat 3
	a.setFlat(3, 7.5f);
	EXPECT_FLOAT_EQ(a.getFlat<float>(3), 7.5f);
	EXPECT_FLOAT_EQ(a.get<float>({1, 0}), 7.5f);

	for (size_t i = 0; i < a.numElements(); ++i)
		a.setFlat(i, (float)i);
	for (size_t i = 0; i < a.numElements(); ++i)
		EXPECT_FLOAT_EQ(a.getFlat<float>(i), (float)i);
}

TEST(NDArray_indexing, Nested_OperatorBrackets) {
	NDArray m({2, 3}, F32);
	m[0][1] = 3.14f;
	m[1][2] = 2.71f;
	float a = m[0][1];
	float b = m[1][2];
	EXPECT_FLOAT_EQ(a, 3.14f);
	EXPECT_FLOAT_EQ(b, 2.71f);

	const NDArray& cm = m;
	float c = cm[0][1];
	EXPECT_FLOAT_EQ(c, 3.14f);
}

TEST(NDArray_indexing, RankMismatch_Throws) {
	NDArray m({2, 3}, F32);
	ArrayList<int> bad;
	bad.add(0);
	EXPECT_THROW(m.get<float>(bad), std::out_of_range);
	EXPECT_THROW(m.getFlat<float>(100), std::out_of_range);
}

TEST(NDArray_indexing, Scalar_EmptyIndices) {
	NDArray s(F32, 1.5f);
	ArrayList<int> empty;
	EXPECT_FLOAT_EQ(s.get<float>(empty), 1.5f);
	s.set(empty, 2.5f);
	EXPECT_FLOAT_EQ(s.get<float>({}), 2.5f);
}
