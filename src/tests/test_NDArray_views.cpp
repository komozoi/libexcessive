
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-2
//
// All rights reserved.

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#if defined(__APPLE__)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#include "NDArray.h"
#include "alloc/pointer.h"

static size_t heapInUse() {
#if defined(__APPLE__)
	malloc_statistics_t st;
	malloc_zone_statistics(malloc_default_zone(), &st);
	return (size_t)st.size_in_use;
#elif defined(__GLIBC__)
	struct mallinfo2 mi = mallinfo2();
	return (size_t)mi.uordblks;
#else
	struct mallinfo mi = mallinfo();
	return (size_t)mi.uordblks;
#endif
}


TEST(NDArray_views, DefaultView_IsEmpty) {
	NDArrayView v;
	EXPECT_EQ(v.numElements(), (size_t)0);
	EXPECT_EQ(v.getShape().size(), 1);
	EXPECT_EQ(v.getShape().get(0), 0);
	EXPECT_EQ(v.sharedBuffer().get(), nullptr);
	EXPECT_THROW(v.getFlat<float>(0), std::out_of_range);
}

TEST(NDArray_views, GetFlat_INT64_PreservesAbove2p53) {
	const int64_t big = (int64_t)1 << 60;
	NDArray a({2}, INT64);
	a.set({0}, big);
	a.set({1}, -big);
	EXPECT_EQ(a.getFlat<int64_t>(0), big);
	EXPECT_EQ(a.view().getFlat<int64_t>(0), big);
	EXPECT_EQ(a.view().getFlat<int64_t>(1), -big);
	EXPECT_EQ(a.view().get<int64_t>(ArrayList<int>({0})), big);

	NDArray m({2, 2}, INT64);
	m.set({0, 0}, big);
	m.set({0, 1}, (int64_t)1);
	m.set({1, 0}, (int64_t)2);
	m.set({1, 1}, -big);
	NDArrayView t = m.transpose();
	EXPECT_FALSE(t.isContiguous());
	EXPECT_EQ(t.get<int64_t>(ArrayList<int>({0, 1})), (int64_t)2);
	EXPECT_EQ(t.get<int64_t>(ArrayList<int>({1, 0})), (int64_t)1);
	EXPECT_EQ(t.getFlat<int64_t>(1), (int64_t)2);
	EXPECT_EQ(t.getFlat<int64_t>(2), (int64_t)1);
	EXPECT_EQ(t.get<int64_t>(ArrayList<int>({1, 1})), -big);
}

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

TEST(NDArray_views, Reshape_OwnedSharesBuffer) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
	NDArray b = a.reshape(ArrayList<int>({2, 3}));
	EXPECT_EQ(b.shape.get(0), 2);
	EXPECT_EQ(b.shape.get(1), 3);
	EXPECT_EQ(static_cast<const NDArray&>(b).data(),
	          static_cast<const NDArray&>(a).data());
	EXPECT_FLOAT_EQ(b.get<float>({1, 2}), 6.0f);
	b.set({0, 0}, 99.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
	EXPECT_THROW(a.reshape(ArrayList<int>({2, 2})), std::invalid_argument);
}

TEST(NDArray_views, ReshapeOwned_OffsetCopies) {
	NDArray m({2, 3}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
	NDArrayView row = m.row(1);
	NDArray c = row.reshapeOwned(ArrayList<int>({3}));
	EXPECT_TRUE(c.ownsStorage());
	EXPECT_NE(c.data(), m.data());
	EXPECT_FLOAT_EQ(c.get<float>({0}), 4.0f);
	EXPECT_FLOAT_EQ(c.get<float>({2}), 6.0f);
	NDArray shared = m.view().reshapeOwned(ArrayList<int>({3, 2}));
	EXPECT_EQ(static_cast<const NDArray&>(shared).data(),
	          static_cast<const NDArray&>(m).data());
}

TEST(NDArray_views, Transpose_F32_And_INT3) {
	NDArray a({2, 3}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
	NDArrayView t = a.transpose();
	EXPECT_EQ(t.getShape().get(0), 3);
	EXPECT_EQ(t.getShape().get(1), 2);
	EXPECT_FALSE(t.isContiguous());
	EXPECT_FLOAT_EQ(t.get<float>(ArrayList<int>({0, 1})), 4.0f);
	EXPECT_FLOAT_EQ(t.get<float>(ArrayList<int>({2, 0})), 3.0f);
	EXPECT_FLOAT_EQ(t.get<float>(ArrayList<int>({2, 1})), 6.0f);

	NDArray i3({2, 3}, INT3);
	const int vals[6] = {-4, -1, 0, 1, 2, 3};
	for (int k = 0; k < 6; ++k)
		i3.setFlat((size_t)k, vals[k]);
	NDArrayView it = i3.transpose();
	EXPECT_EQ(it.get<int>(ArrayList<int>({0, 1})), 1);
	EXPECT_EQ(it.get<int>(ArrayList<int>({2, 0})), 0);
	EXPECT_EQ(it.get<int>(ArrayList<int>({2, 1})), 3);
}

TEST(NDArray_views, Swapaxes_And_Permute_Rank3) {
	NDArray a({2, 3, 4}, F32);
	float v = 0;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 4; ++k)
				a.set({i, j, k}, v++);
	NDArrayView s = a.swapaxes(0, 2);
	EXPECT_EQ(s.getShape().get(0), 4);
	EXPECT_EQ(s.getShape().get(2), 2);
	EXPECT_FLOAT_EQ(s.get<float>(ArrayList<int>({3, 1, 0})), a.get<float>({0, 1, 3}));

	ArrayList<int> axes;
	axes.add(1);
	axes.add(2);
	axes.add(0);
	NDArrayView p = a.permute(axes);
	EXPECT_EQ(p.getShape().get(0), 3);
	EXPECT_EQ(p.getShape().get(1), 4);
	EXPECT_EQ(p.getShape().get(2), 2);
	EXPECT_FLOAT_EQ(p.get<float>(ArrayList<int>({2, 3, 1})), a.get<float>({1, 2, 3}));

	ArrayList<int> bad;
	bad.add(0);
	bad.add(0);
	bad.add(1);
	EXPECT_THROW(a.permute(bad), std::invalid_argument);
}

TEST(NDArray_views, RowColSlice_Rank2And3) {
	NDArray m({2, 3}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
	NDArrayView r1 = m.row(1);
	EXPECT_EQ(r1.getShape().size(), 1);
	EXPECT_EQ(r1.getShape().get(0), 3);
	EXPECT_TRUE(r1.isContiguous());
	EXPECT_FLOAT_EQ(r1.getFlat<float>(0), 4.0f);
	EXPECT_FLOAT_EQ(r1.getFlat<float>(2), 6.0f);
	alignas(4) float raw[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	NDArray w = NDArray::wrap(raw, sizeof(raw), {2, 3}, F32);
	NDArrayView wr = w.row(1);
	w.set({1, 1}, 50.0f);
	EXPECT_FLOAT_EQ(wr.getFlat<float>(1), 50.0f);

	NDArrayView c0 = m.col(0);
	EXPECT_EQ(c0.getShape().size(), 1);
	EXPECT_EQ(c0.getShape().get(0), 2);
	EXPECT_FALSE(c0.isContiguous());
	EXPECT_FLOAT_EQ(c0.getFlat<float>(0), 1.0f);
	EXPECT_FLOAT_EQ(c0.getFlat<float>(1), 4.0f);

	NDArray t({2, 3, 4}, F32);
	float x = 0;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 4; ++k)
				t.set({i, j, k}, x++);
	NDArrayView plane = t.row(1);
	EXPECT_EQ(plane.getShape().size(), 2);
	EXPECT_EQ(plane.getShape().get(0), 3);
	EXPECT_EQ(plane.getShape().get(1), 4);
	EXPECT_FLOAT_EQ(plane.get<float>(ArrayList<int>({0, 0})), t.get<float>({1, 0, 0}));
	EXPECT_FLOAT_EQ(plane.get<float>(ArrayList<int>({2, 3})), t.get<float>({1, 2, 3}));
	EXPECT_THROW(m.row(2), std::out_of_range);
	EXPECT_THROW(m.slice(-1, 0), std::out_of_range);
}

TEST(NDArray_views, RangeSlice_Rank2_F32) {
	NDArray m({4, 3}, ArrayList({
		0.0f, 1.0f, 2.0f,
		3.0f, 4.0f, 5.0f,
		6.0f, 7.0f, 8.0f,
		9.0f, 10.0f, 11.0f
	}));
	NDArrayView v = m.slice(0, 1, 3);
	EXPECT_EQ(v.getShape().size(), 2);
	EXPECT_EQ(v.getShape().get(0), 2);
	EXPECT_EQ(v.getShape().get(1), 3);
	EXPECT_TRUE(v.isContiguous());
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({0, 0})), 3.0f);
	EXPECT_FLOAT_EQ(v.get<float>(ArrayList<int>({1, 2})), 8.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(0), 3.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(5), 8.0f);
	NDArray c = v.copy();
	EXPECT_EQ(c.shape.get(0), 2);
	EXPECT_EQ(c.shape.get(1), 3);
	EXPECT_FLOAT_EQ(c.getFlat<float>(0), 3.0f);
	EXPECT_FLOAT_EQ(c.getFlat<float>(5), 8.0f);

	NDArrayView cols = m.slice(1, 1, 3);
	EXPECT_EQ(cols.getShape().get(0), 4);
	EXPECT_EQ(cols.getShape().get(1), 2);
	EXPECT_FALSE(cols.isContiguous());
	EXPECT_FLOAT_EQ(cols.get<float>(ArrayList<int>({0, 0})), 1.0f);
	EXPECT_FLOAT_EQ(cols.get<float>(ArrayList<int>({2, 1})), 8.0f);
}

TEST(NDArray_views, RangeSlice_EmptyAndFull) {
	NDArray m({4, 3}, F32);
	for (int i = 0; i < 12; ++i)
		m.setFlat((size_t)i, (float)i);
	NDArrayView empty = m.slice(0, 2, 2);
	EXPECT_EQ(empty.getShape().size(), 2);
	EXPECT_EQ(empty.getShape().get(0), 0);
	EXPECT_EQ(empty.getShape().get(1), 3);
	EXPECT_EQ(empty.numElements(), (size_t)0);
	EXPECT_THROW(empty.getFlat<float>(0), std::out_of_range);

	NDArrayView full = m.slice(0, 0, 4);
	EXPECT_EQ(full.getShape().get(0), 4);
	EXPECT_EQ(full.getShape().get(1), 3);
	EXPECT_EQ(full.getOffset(), m.view().getOffset());
	for (int i = 0; i < 12; ++i)
		EXPECT_FLOAT_EQ(full.getFlat<float>((size_t)i), m.getFlat<float>((size_t)i));
}

TEST(NDArray_views, RangeSlice_BoundsAndIndexSliceUnchanged) {
	NDArray m({2, 3}, ArrayList({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
	EXPECT_THROW(m.slice(-1, 0, 1), std::out_of_range);
	EXPECT_THROW(m.slice(2, 0, 1), std::out_of_range);
	EXPECT_THROW(m.slice(0, -1, 1), std::out_of_range);
	EXPECT_THROW(m.slice(0, 0, 3), std::out_of_range);
	EXPECT_THROW(m.slice(0, 2, 1), std::out_of_range);
	NDArray s(F32, 1.0f);
	EXPECT_THROW(s.slice(0, 0, 1), std::invalid_argument);

	NDArrayView dropped = m.slice(0, 1);
	EXPECT_EQ(dropped.getShape().size(), 1);
	EXPECT_EQ(dropped.getShape().get(0), 3);
	EXPECT_FLOAT_EQ(dropped.getFlat<float>(0), 4.0f);
}

TEST(NDArray_views, RangeSlice_INT3) {
	NDArray a({2, 4}, INT3);
	const int vals[8] = {-4, -1, 0, 1, 2, 3, -2, -3};
	for (int k = 0; k < 8; ++k)
		a.setFlat((size_t)k, vals[k]);
	NDArrayView v = a.slice(1, 1, 3);
	EXPECT_EQ(v.getShape().size(), 2);
	EXPECT_EQ(v.getShape().get(0), 2);
	EXPECT_EQ(v.getShape().get(1), 2);
	EXPECT_EQ(v.get<int>(ArrayList<int>({0, 0})), -1);
	EXPECT_EQ(v.get<int>(ArrayList<int>({0, 1})), 0);
	EXPECT_EQ(v.get<int>(ArrayList<int>({1, 0})), 3);
	EXPECT_EQ(v.get<int>(ArrayList<int>({1, 1})), -2);
	NDArray c = v.copy();
	EXPECT_EQ(c.type, INT3);
	EXPECT_EQ(c.getFlat<int>(0), -1);
	EXPECT_EQ(c.getFlat<int>(3), -2);
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

// Deterministic view / stride / wrap fuzz (issue 045).
static const int kFuzzMaxR = 6;

struct FuzzRng {
	uint32_t s;
	uint32_t next() {
		s = s * 1664525u + 1013904223u;
		return s;
	}
	int pick(int lo, int hi) {
		return lo + (int)(next() % (uint32_t)(hi - lo));
	}
};

struct FuzzGeom {
	int rank;
	int shape[kFuzzMaxR];
	int origAxis[kFuzzMaxR];
	int origRank;
	int origShape[kFuzzMaxR];
	int origFixed[kFuzzMaxR];
};

static void fuzzGeomInit(FuzzGeom& g, const ArrayList<int>& sh) {
	g.origRank = sh.size();
	g.rank = g.origRank;
	for (int d = 0; d < g.rank; ++d) {
		g.shape[d] = sh.get(d);
		g.origAxis[d] = d;
		g.origShape[d] = sh.get(d);
		g.origFixed[d] = -1;
	}
}

static void fuzzGeomSlice(FuzzGeom& g, int axis, int index) {
	int oa = g.origAxis[axis];
	if (oa >= 0)
		g.origFixed[oa] = index;
	for (int d = axis; d < g.rank - 1; ++d) {
		g.shape[d] = g.shape[d + 1];
		g.origAxis[d] = g.origAxis[d + 1];
	}
	--g.rank;
}

static void fuzzGeomTranspose(FuzzGeom& g) {
	if (g.rank < 2)
		return;
	for (int i = 0, j = g.rank - 1; i < j; ++i, --j) {
		int ts = g.shape[i];
		g.shape[i] = g.shape[j];
		g.shape[j] = ts;
		int ta = g.origAxis[i];
		g.origAxis[i] = g.origAxis[j];
		g.origAxis[j] = ta;
	}
}

static void fuzzGeomSwap(FuzzGeom& g, int a, int b) {
	int ts = g.shape[a];
	g.shape[a] = g.shape[b];
	g.shape[b] = ts;
	int ta = g.origAxis[a];
	g.origAxis[a] = g.origAxis[b];
	g.origAxis[b] = ta;
}

static void fuzzGeomBroadcast(FuzzGeom& g, const ArrayList<int>& target) {
	const int tr = target.size();
	int newShape[kFuzzMaxR];
	int newOrig[kFuzzMaxR];
	for (int i = 0; i < tr; ++i) {
		int src = i - (tr - g.rank);
		newShape[i] = target.get(i);
		newOrig[i] = src < 0 ? -1 : g.origAxis[src];
	}
	g.rank = tr;
	for (int i = 0; i < tr; ++i) {
		g.shape[i] = newShape[i];
		g.origAxis[i] = newOrig[i];
	}
}

static ArrayList<int> fuzzOrigIdx(const FuzzGeom& g, const int* viewCoords) {
	int orig[kFuzzMaxR];
	for (int d = 0; d < g.origRank; ++d)
		orig[d] = g.origFixed[d] >= 0 ? g.origFixed[d] : 0;
	for (int d = 0; d < g.rank; ++d) {
		int oa = g.origAxis[d];
		if (oa < 0 || g.origFixed[oa] >= 0)
			continue;
		orig[oa] = g.origShape[oa] <= 1 ? 0 : viewCoords[d];
	}
	ArrayList<int> idx;
	for (int d = 0; d < g.origRank; ++d)
		idx.add(orig[d]);
	return idx;
}

static void fuzzFlatToCoords(const FuzzGeom& g, size_t flat, int* coords) {
	size_t r = flat;
	for (int d = g.rank - 1; d >= 0; --d) {
		int dim = g.shape[d];
		coords[d] = dim > 0 ? (int)(r % (size_t)dim) : 0;
		if (dim > 0)
			r /= (size_t)dim;
	}
}

static void fuzzCheckView(const NDArray& owner, const NDArrayView& v, const FuzzGeom& g) {
	ASSERT_EQ(v.getShape().size(), g.rank);
	for (int d = 0; d < g.rank; ++d)
		ASSERT_EQ(v.getShape().get(d), g.shape[d]);
	const size_t n = v.numElements();
	if (n == 0)
		return;
	NDArray copied = v.copy();
	ASSERT_EQ(copied.numElements(), n);
	for (size_t i = 0; i < n; ++i) {
		int coords[kFuzzMaxR];
		fuzzFlatToCoords(g, i, coords);
		ArrayList<int> viewIdx;
		for (int d = 0; d < g.rank; ++d)
			viewIdx.add(coords[d]);
		ArrayList<int> srcIdx = fuzzOrigIdx(g, coords);
		switch (owner.type) {
			case F32: {
				float exp = owner.get<float>(srcIdx);
				EXPECT_FLOAT_EQ(v.getFlat<float>(i), exp);
				EXPECT_FLOAT_EQ(v.get<float>(viewIdx), exp);
				EXPECT_FLOAT_EQ(copied.getFlat<float>(i), exp);
				break;
			}
			case INT32: {
				int32_t exp = owner.get<int32_t>(srcIdx);
				EXPECT_EQ(v.getFlat<int32_t>(i), exp);
				EXPECT_EQ(v.get<int32_t>(viewIdx), exp);
				EXPECT_EQ(copied.getFlat<int32_t>(i), exp);
				break;
			}
			case UINT8: {
				uint8_t exp = owner.get<uint8_t>(srcIdx);
				EXPECT_EQ(v.getFlat<uint8_t>(i), exp);
				EXPECT_EQ(v.get<uint8_t>(viewIdx), exp);
				EXPECT_EQ(copied.getFlat<uint8_t>(i), exp);
				break;
			}
			case INT3: {
				int exp = owner.get<int>(srcIdx);
				EXPECT_EQ(v.getFlat<int>(i), exp);
				EXPECT_EQ(v.get<int>(viewIdx), exp);
				EXPECT_EQ(copied.getFlat<int>(i), exp);
				break;
			}
			default:
				FAIL() << "unexpected fuzz type";
		}
	}
}

TEST(NDArray_views, Fuzz_SliceTransposeBroadcastWrap) {
	FuzzRng rng;
	rng.s = 0xC0FFEE01u;
	const NDArrayType types[4] = { F32, INT32, UINT8, INT3 };
	for (int trial = 0; trial < 160; ++trial) {
		const int rank = rng.pick(1, 4);
		ArrayList<int> shape;
		int nElem = 1;
		for (int d = 0; d < rank; ++d) {
			int dim = rng.pick(1, 5);
			shape.add(dim);
			nElem *= dim;
		}
		if (nElem > 80)
			continue;
		NDArrayType ty = types[rng.pick(0, 4)];
		NDArray owner(shape, ty);
		for (int i = 0; i < nElem; ++i) {
			if (ty == F32)
				owner.setFlat((size_t)i, (float)i);
			else if (ty == INT32)
				owner.setFlat((size_t)i, (int32_t)i);
			else if (ty == UINT8)
				owner.setFlat((size_t)i, (int)(i & 255));
			else
				owner.setFlat((size_t)i, (i % 8) - 4);
		}

		FuzzGeom g;
		fuzzGeomInit(g, shape);
		NDArrayView v = owner.view();
		if ((trial & 1) != 0 && (ty == F32 || ty == INT32 || ty == UINT8)) {
			v = NDArrayView::wrap(owner.data(), owner.byteSize(), shape, ty);
			EXPECT_EQ(v.data(), owner.data());
		}

		const int nOps = rng.pick(1, 5);
		for (int step = 0; step < nOps; ++step) {
			if (g.rank == 0)
				break;
			int kind = rng.pick(0, 5);
			if (kind == 0 && g.rank >= 1) {
				int axis = rng.pick(0, g.rank);
				if (g.shape[axis] <= 0)
					continue;
				int index = rng.pick(0, g.shape[axis]);
				v = v.slice(axis, index);
				fuzzGeomSlice(g, axis, index);
			} else if (kind == 1) {
				v = v.transpose();
				fuzzGeomTranspose(g);
			} else if (kind == 2 && g.rank >= 2) {
				int a = rng.pick(0, g.rank);
				int b = rng.pick(0, g.rank);
				if (a == b)
					continue;
				v = v.swapaxes(a, b);
				fuzzGeomSwap(g, a, b);
			} else if (kind == 3 && g.rank >= 1 && g.rank < 4) {
				ArrayList<int> target;
				target.add(2);
				for (int d = 0; d < g.rank; ++d)
					target.add(g.shape[d]);
				if (!v.isBroadcastableTo(target))
					continue;
				v = v.broadcastTo(target);
				fuzzGeomBroadcast(g, target);
			} else if (kind == 4 && g.rank >= 2) {
				int last = g.rank - 1;
				if (g.shape[last] <= 0)
					continue;
				int index = rng.pick(0, g.shape[last]);
				v = v.col(index);
				fuzzGeomSlice(g, last, index);
			}
		}
		fuzzCheckView(owner, v, g);
	}
}

TEST(NDArray_views, Wrap_RetainedMatchesEmptyHeap) {
	const int n = 20000;
	float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	ArrayList<int> sh({4});

	std::vector<NDArrayView> emptyViews;
	emptyViews.reserve((size_t)n);
	const size_t hEmpty0 = heapInUse();
	for (int i = 0; i < n; ++i)
		emptyViews.emplace_back();
	const size_t emptyCost = heapInUse() - hEmpty0;

	std::vector<NDArrayView> wraps;
	wraps.reserve((size_t)n);
	const size_t hWrap0 = heapInUse();
	for (int i = 0; i < n; ++i)
		wraps.push_back(NDArrayView::wrap(buf, sizeof(buf), sh, F32));
	const size_t wrapCost = heapInUse() - hWrap0;

	ASSERT_EQ(wraps[(size_t)n - 1].getFlat<float>(0), 1.0f);
	EXPECT_LE(wrapCost, emptyCost + (size_t)4096);
}

TEST(NDArray_views, Slice_RetainedNoExtraHeap) {
	const int rows = 4000;
	NDArray a({rows, 8}, F32);
	for (int i = 0; i < rows * 8; ++i)
		a.setFlat((size_t)i, (float)i);
	NDArrayView owner = a.view();

	std::vector<NDArrayView> slices;
	slices.reserve((size_t)rows);
	const size_t h0 = heapInUse();
	for (int i = 0; i < rows; ++i)
		slices.push_back(owner.row(i));
	const size_t cost = heapInUse() - h0;

	EXPECT_FLOAT_EQ(slices[1].getFlat<float>(0), 8.0f);
	EXPECT_LE(cost, (size_t)4096);
}

TEST(NDArray_views, RankAboveMax_Throws) {
	ArrayList<int> sh;
	for (int i = 0; i < NDArray::kMaxRank + 1; ++i)
		sh.add(1);
	float buf[1] = {1.0f};
	EXPECT_THROW(NDArrayView::wrap(buf, sizeof(buf), sh, F32), std::out_of_range);
	NDArray a(sh, F32);
	EXPECT_THROW(a.view(), std::out_of_range);
}
