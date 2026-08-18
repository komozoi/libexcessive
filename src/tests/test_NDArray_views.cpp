
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.

#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>

#include "NDArray.h"
#include "alloc/pointer.h"


TEST(NDArray_views, View_SharesData) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArrayView v = a.view();
	EXPECT_EQ(v.getShape().size(), 1);
	EXPECT_EQ(v.getShape().get(0), 3);
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
	EXPECT_EQ(v.getShape().get(0), 2);
	EXPECT_EQ(v.getShape().get(1), 3);
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
	EXPECT_EQ(v.getShape().get(0), 2);
	EXPECT_EQ(v.getShape().get(1), 3);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({1, 2})), 6.0f);
}

TEST(NDArray_views, NotBroadcastable_Throws) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	EXPECT_FALSE(a.isBroadcastableTo(ArrayList<int>({2, 4})));
	EXPECT_THROW(a.broadcastTo(ArrayList<int>({2, 4})), std::invalid_argument);
}

TEST(NDArray_views, CannotDropLeadingAxis) {
	NDArray a({2, 3}, F32);
	EXPECT_FALSE(a.isBroadcastableTo(ArrayList<int>({3})));
	EXPECT_THROW(a.broadcastTo(ArrayList<int>({3})), std::invalid_argument);
}

TEST(NDArray_views, ReshapeNegativeAxis_Throws) {
	NDArray empty({0}, F32);
	EXPECT_THROW(empty.reshapeView(ArrayList<int>({-1})), std::invalid_argument);
}

TEST(NDArray_views, ReshapeProductOverflow_Throws) {
	// wrapped product is 0, matching an empty source — must still reject
	NDArray empty({0}, F32);
	EXPECT_THROW(empty.reshapeView(ArrayList<int>({1 << 30, 1 << 30, 16})),
	             std::invalid_argument);
}

TEST(NDArray_views, BroadcastNegativeAxis_Throws) {
	NDArray a({1}, F32);
	EXPECT_THROW(a.broadcastTo(ArrayList<int>({-1, 8})), std::invalid_argument);
}

TEST(NDArray_views, BroadcastProductOverflow_Throws) {
	NDArray s({}, F32);
	EXPECT_THROW(s.broadcastTo(ArrayList<int>({1 << 30, 1 << 30, 1 << 10})),
	             std::invalid_argument);
}

TEST(NDArray_views, RowSlice_OffsetNonzero_IsContiguous) {
	NDArray m({3, 4}, ArrayList({
		0.0f, 1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f, 7.0f,
		8.0f, 9.0f, 10.0f, 11.0f
	}));
	NDArrayView row(m.view().sharedBuffer(), ArrayList<int>({4}),
	                ArrayList<size_t>({(size_t)1}), 4, F32);
	EXPECT_TRUE(row.isContiguous());
	EXPECT_EQ(row.getOffset(), 4u);
	EXPECT_FLOAT_EQ(row.getFlat<float>(0), 4.0f);
	EXPECT_FLOAT_EQ(row.getFlat<float>(3), 7.0f);

	NDArrayView reshaped = row.reshape(ArrayList<int>({2, 2}));
	EXPECT_TRUE(reshaped.isContiguous());
	EXPECT_EQ(reshaped.getOffset(), 4u);
	EXPECT_FLOAT_EQ(reshaped.get<float>(ArrayList<int>({0, 0})), 4.0f);
	EXPECT_FLOAT_EQ(reshaped.get<float>(ArrayList<int>({1, 1})), 7.0f);
}

TEST(NDArray_views, ContiguousCopy_MatchesOwnerBytes) {
	NDArray m({2, 4}, ArrayList({
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f
	}));
	NDArrayView row(m.view().sharedBuffer(), ArrayList<int>({4}),
	                ArrayList<size_t>({(size_t)1}), 4, F32);
	NDArray c = row.copy();
	EXPECT_TRUE(c.ownsStorage());
	EXPECT_EQ(c.byteSize(), 4 * sizeof(float));
	EXPECT_EQ(memcmp(c.data(), (const float*)m.data() + 4, 4 * sizeof(float)), 0);
	EXPECT_FLOAT_EQ(c.getFlat<float>(0), 5.0f);
	EXPECT_FLOAT_EQ(c.getFlat<float>(3), 8.0f);
	c.setFlat(0, 99.0f);
	EXPECT_FLOAT_EQ(m.get<float>({1, 0}), 5.0f);
}

TEST(NDArray_views, ColumnSlice_NotContiguous) {
	NDArray m({2, 3}, ArrayList({
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	}));
	NDArrayView col(m.view().sharedBuffer(), ArrayList<int>({2}),
	                ArrayList<size_t>({(size_t)3}), 1, F32);
	EXPECT_FALSE(col.isContiguous());
	EXPECT_THROW(col.reshape(ArrayList<int>({2})), std::invalid_argument);
	NDArray c = col.copy();
	EXPECT_FLOAT_EQ(c.getFlat<float>(0), 2.0f);
	EXPECT_FLOAT_EQ(c.getFlat<float>(1), 5.0f);
}

TEST(NDArray_views, GetFlat_StridedNoHeapCoords) {
	NDArray m({2, 3}, ArrayList({
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	}));
	NDArrayView col(m.view().sharedBuffer(), ArrayList<int>({2}),
	                ArrayList<size_t>({(size_t)3}), 1, F32);
	EXPECT_FALSE(col.isContiguous());
	EXPECT_FLOAT_EQ(col.getFlat<float>(0), 2.0f);
	EXPECT_FLOAT_EQ(col.getFlat<float>(1), 5.0f);
}
