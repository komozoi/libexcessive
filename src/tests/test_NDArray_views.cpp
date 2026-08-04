
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.

#include <gtest/gtest.h>

#include "NDArray.h"
#include "alloc/pointer.h"


TEST(NDArray_views, View_SharesData) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArrayView v = a.view();
	EXPECT_EQ(v.shape.size(), 1);
	EXPECT_EQ(v.shape.get(0), 3);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({0})), 1.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(2), 3.0f);
}

TEST(NDArray_views, View_OutlivesOwner) {
	NDArrayView v;
	{
		NDArray a(ArrayList({10.0f, 20.0f}));
		v = a.view();
		EXPECT_GE(v.sharedBuffer().numReferences(), 2);
	}
	// owner destroyed; view still valid via CoW refcount
	EXPECT_FLOAT_EQ(v.getFlat<float>(0), 10.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(1), 20.0f);
}

TEST(NDArray_views, CoW_DetachOnWrite) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b = a; // share
	// Mutating a should not change b
	a.set({0}, 99.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 99.0f);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 1.0f);
}

/** Regression: squared() chain must not corrupt earlier results via shared buffers. */
TEST(NDArray_views, CoW_SquaredChain_DoesNotCorrupt) {
	NDArray b(ArrayList({2.0f, 3.0f}));
	NDArray b2 = b.squared();
	NDArray b4 = b2.squared();
	EXPECT_FLOAT_EQ(b.getFlat<float>(0), 2.0f);
	EXPECT_FLOAT_EQ(b2.getFlat<float>(0), 4.0f);
	EXPECT_FLOAT_EQ(b4.getFlat<float>(0), 16.0f);
	EXPECT_FLOAT_EQ(b2.getFlat<float>(1), 9.0f);
	EXPECT_FLOAT_EQ(b4.getFlat<float>(1), 81.0f);
}

TEST(NDArray_views, View_BroadcastThenCopy_Independent) {
	NDArray a(ArrayList({5.0f}));
	NDArrayView v = a.broadcastTo(ArrayList<int>({3}));
	NDArray mat = v.copy();
	a.set({0}, 0.0f);
	// materialised copy keeps old values
	EXPECT_FLOAT_EQ(mat.getFlat<float>(0), 5.0f);
	EXPECT_FLOAT_EQ(mat.getFlat<float>(2), 5.0f);
}

TEST(NDArray_views, BroadcastTo_Trailing) {
	// shape {2,1} broadcast to {2,3}
	NDArray a({2, 1}, ArrayList({10.0f, 20.0f}));
	NDArrayView v = a.broadcastTo(ArrayList<int>({2, 3}));
	EXPECT_EQ(v.shape.get(0), 2);
	EXPECT_EQ(v.shape.get(1), 3);
	// all columns of row 0 are 10
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({0, 0})), 10.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({0, 1})), 10.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({0, 2})), 10.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({1, 0})), 20.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({1, 2})), 20.0f);

	NDArray mat = v.copy();
	EXPECT_EQ(mat.type, F32);
	EXPECT_FLOAT_EQ(mat.get<float>({0, 1}), 10.0f);
	EXPECT_FLOAT_EQ(mat.get<float>({1, 1}), 20.0f);
}

TEST(NDArray_views, BroadcastTo_LeadingOnes) {
	// {3} → {2,3}: right-align, leading dim inserted as broadcast
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	ASSERT_TRUE(a.isBroadcastableTo(ArrayList<int>({2, 3})));
	NDArrayView v = a.broadcastTo(ArrayList<int>({2, 3}));
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({0, 0})), 1.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({1, 0})), 1.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({0, 2})), 3.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({1, 2})), 3.0f);
}

TEST(NDArray_views, BroadcastTo_InsertMiddle) {
	// Explicit shape with size-1: {2,3,1} → {2,3,4}
	NDArray a({2, 3, 1}, F32);
	float v = 0;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			a.set({i, j, 0}, v++);
	NDArrayView br = a.broadcastTo(ArrayList<int>({2, 3, 4}));
	EXPECT_FLOAT_EQ(br.get<float>(ArrayList<int>({0, 0, 0})), 0.0f);
	EXPECT_FLOAT_EQ(br.get<float>(ArrayList<int>({0, 0, 3})), 0.0f); // broadcast
	EXPECT_FLOAT_EQ(br.get<float>(ArrayList<int>({1, 2, 1})), 5.0f);
}

TEST(NDArray_views, ReshapeView) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
	NDArrayView v = a.reshapeView(ArrayList<int>({2, 3}));
	EXPECT_EQ(v.shape.get(0), 2);
	EXPECT_EQ(v.shape.get(1), 3);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({1, 2})), 6.0f);
}

TEST(NDArray_views, NotBroadcastable_Throws) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	EXPECT_FALSE(a.isBroadcastableTo(ArrayList<int>({2, 4})));
	EXPECT_THROW(a.broadcastTo(ArrayList<int>({2, 4})), std::invalid_argument);
}
