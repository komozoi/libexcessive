
// Copyright 2021-2026 komozoi
// Original Creation Date: 2026-8-17
//
// All rights reserved.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

#include "NDArray.h"


static void expectEqualPacked(const NDArrayView& v, const NDArray& owned) {
	ASSERT_EQ(v.getType(), owned.type);
	ASSERT_EQ(v.getShape().size(), owned.shape.size());
	for (int i = 0; i < v.getShape().size(); ++i)
		ASSERT_EQ(v.getShape().get(i), owned.shape.get(i));
	const size_t n = owned.numElements();
	ASSERT_EQ(v.numElements(), n);
	for (size_t i = 0; i < n; ++i) {
		switch (owned.type) {
			case F32:
				EXPECT_FLOAT_EQ(v.getFlat<float>(i), owned.getFlat<float>(i));
				break;
			case UINT8:
				EXPECT_EQ(v.getFlat<uint8_t>(i), owned.getFlat<uint8_t>(i));
				break;
			case INT8:
				EXPECT_EQ(v.getFlat<int8_t>(i), owned.getFlat<int8_t>(i));
				break;
			case INT3:
				EXPECT_EQ(v.getFlat<int>(i), owned.getFlat<int>(i));
				break;
			case BINARY:
				EXPECT_EQ(v.getFlat<int>(i), owned.getFlat<int>(i));
				break;
			default:
				FAIL() << "unexpected type in expectEqualPacked";
		}
	}
}


TEST(NDArray_wrap, F16_BitsRoundTrip) {
	alignas(2) uint16_t bits[3] = {
		ndarray_half::f32ToF16(1.0f),
		ndarray_half::f32ToF16(-2.0f),
		ndarray_half::f32ToF16(0.5f)
	};
	NDArray w = NDArray::wrap(bits, sizeof(bits), {3}, F16);
	EXPECT_EQ(w.data(), (void*)bits);
	EXPECT_FLOAT_EQ(w.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(w.get<float>({1}), -2.0f);
	EXPECT_FLOAT_EQ(w.get<float>({2}), 0.5f);
	w.set({0}, 4.0f);
	EXPECT_EQ(bits[0], ndarray_half::f32ToF16(4.0f));
}

TEST(NDArray_wrap, DataReturnsCallerPointer) {
	float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	NDArray w = NDArray::wrap(buf, sizeof(buf), {4}, F32);
	ASSERT_EQ(w.data(), (void*)buf);
	EXPECT_EQ(w.byteSize(), sizeof(buf));
	EXPECT_EQ(w.view().data(), (const void*)buf);
	EXPECT_EQ(w.view().byteSize(), sizeof(buf));

	static_cast<float*>(w.data())[1] = 8.0f;
	EXPECT_FLOAT_EQ(buf[1], 8.0f);
	EXPECT_FLOAT_EQ(w.getFlat<float>(1), 8.0f);
}

TEST(NDArray_wrap, View_F32_StackMatchesOwned) {
	float buf[4] = {1.5f, -2.0f, 3.25f, 0.0f};
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {4}, F32);
	NDArray owned(ArrayList({1.5f, -2.0f, 3.25f, 0.0f}));
	expectEqualPacked(v, owned);
	EXPECT_FALSE(v.sharedBuffer().get()->ownsData);
}

TEST(NDArray_wrap, View_UINT8_StackMatchesOwned) {
	uint8_t buf[5] = {0, 1, 255, 16, 7};
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {5}, UINT8);
	NDArray owned({5}, UINT8);
	for (int i = 0; i < 5; ++i)
		owned.set({i}, buf[i]);
	expectEqualPacked(v, owned);
}

TEST(NDArray_wrap, View_INT8_StackMatchesOwned) {
	int8_t buf[4] = {-128, -1, 0, 127};
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {4}, INT8);
	NDArray owned({4}, INT8);
	for (int i = 0; i < 4; ++i)
		owned.set({i}, buf[i]);
	expectEqualPacked(v, owned);
	EXPECT_EQ(v.getFlat<int>(0), -128);
	EXPECT_EQ(v.getFlat<int>(1), -1);
}

TEST(NDArray_wrap, View_INT3_MatchesCopy) {
	alignas(8) uint8_t storage[8] = {};
	NDArray w = NDArray::wrap(storage, sizeof(storage), {8}, INT3);
	for (int i = 0; i < 8; ++i)
		w.set({i}, i - 4);
	NDArrayView v = NDArrayView::wrap(storage, sizeof(storage), {8}, INT3);
	NDArray owned = v.copy();
	EXPECT_EQ(owned.type, INT3);
	EXPECT_TRUE(owned.ownsStorage());
	expectEqualPacked(v, owned);
	for (int i = 0; i < 8; ++i)
		EXPECT_EQ(v.get<int>({i}), i - 4);
}

TEST(NDArray_wrap, View_BINARY_MatchesCopy) {
	alignas(8) uint8_t storage[8] = {};
	NDArray w = NDArray::wrap(storage, sizeof(storage), {16}, BINARY);
	for (int i = 0; i < 16; ++i)
		w.set({i}, i % 3 == 0 ? 1 : 0);
	NDArrayView v = NDArrayView::wrap(storage, sizeof(storage), {16}, BINARY);
	NDArray owned = v.copy();
	EXPECT_EQ(owned.type, BINARY);
	expectEqualPacked(v, owned);
	EXPECT_EQ(v.get<int>({0}), 1);
	EXPECT_EQ(v.get<int>({1}), 0);
	EXPECT_EQ(v.get<int>({3}), 1);
}

TEST(NDArray_wrap, DestroyView_DoesNotFreeExternal) {
	auto* buf = new float[3]{1.0f, 2.0f, 3.0f};
	{
		NDArrayView v = NDArrayView::wrap(buf, 3 * sizeof(float), {3}, F32);
		EXPECT_FLOAT_EQ(v.getFlat<float>(1), 2.0f);
	}
	EXPECT_FLOAT_EQ(buf[0], 1.0f);
	EXPECT_FLOAT_EQ(buf[2], 3.0f);
	buf[1] = 9.0f;
	EXPECT_FLOAT_EQ(buf[1], 9.0f);
	delete[] buf;
}

TEST(NDArray_wrap, Mutable_WritesThrough_NoCowDetach) {
	float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	NDArray a = NDArray::wrap(buf, sizeof(buf), {4}, F32);
	EXPECT_FALSE(a.ownsStorage());

	NDArray alias = a; // share wrap
	NDArrayView v = a.view();
	a.set({0}, 99.0f);
	alias.set({1}, 77.0f);

	EXPECT_FLOAT_EQ(buf[0], 99.0f);
	EXPECT_FLOAT_EQ(buf[1], 77.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 99.0f);
	EXPECT_FLOAT_EQ(alias.get<float>({1}), 77.0f);
	EXPECT_FLOAT_EQ(v.get<float>({0}), 99.0f);
	EXPECT_FALSE(a.ownsStorage());
	EXPECT_FALSE(alias.ownsStorage());
}

TEST(NDArray_wrap, ConstBuffer_IsAView) {
	const float buf[3] = {1.0f, 2.0f, 3.0f};
	// NDArray::wrap takes void* — const memory is NDArrayView only.
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {3}, F32);
	EXPECT_FLOAT_EQ(v.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(2), 3.0f);
	EXPECT_FALSE(v.sharedBuffer().get()->ownsData);
}

TEST(NDArray_wrap, ConstReshapeOwned_DoesNotWriteThrough) {
	const float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {2, 2}, F32);
	NDArray a = v.reshapeOwned(ArrayList<int>({4}));
	EXPECT_TRUE(a.ownsStorage());
	a.set({0}, 99.0f);
	EXPECT_FLOAT_EQ(buf[0], 1.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 99.0f);
	EXPECT_FLOAT_EQ(a.get<float>({3}), 4.0f);
}

TEST(NDArray_wrap, UndersizedBuffer_Throws) {
	float buf[2] = {1.0f, 2.0f};
	EXPECT_THROW(NDArrayView::wrap(buf, sizeof(buf), {3}, F32), std::invalid_argument);
	EXPECT_THROW(NDArray::wrap(static_cast<void*>(buf), sizeof(buf), {3}, F32),
	             std::invalid_argument);
}

TEST(NDArray_wrap, NegativeAxis_Throws) {
	float buf[8] = {};
	EXPECT_THROW(NDArrayView::wrap(buf, sizeof(buf), {-1, 8}, F32), std::invalid_argument);
	EXPECT_THROW(NDArray::wrap(static_cast<void*>(buf), sizeof(buf), {-1, 8}, F32),
	             std::invalid_argument);
}

TEST(NDArray_wrap, Uint256ByteSizeOverflow_Throws) {
	alignas(8) uint8_t buf[8] = {};
	EXPECT_THROW(NDArrayView::wrap(buf, sizeof(buf), {1 << 30, 1 << 30}, UINT256),
	             std::invalid_argument);
	EXPECT_THROW(NDArray::wrap(static_cast<void*>(buf), sizeof(buf), {1 << 30, 1 << 30}, UINT256),
	             std::invalid_argument);
}

TEST(NDArray_wrap, ZeroAxis_EmptyOk) {
	NDArrayView v = NDArrayView::wrap(nullptr, 0, {0, 8}, F32);
	EXPECT_EQ(v.numElements(), (size_t)0);
	NDArray w = NDArray::wrap(static_cast<void*>(nullptr), 0, {0, 8}, F32);
	EXPECT_EQ(w.numElements(), (size_t)0);
}

TEST(NDArray_wrap, NullData_Throws) {
	EXPECT_THROW(NDArrayView::wrap(nullptr, 16, {4}, F32), std::invalid_argument);
}

TEST(NDArray_wrap, UnalignedPacked_Throws) {
	alignas(8) uint8_t raw[16] = {};
	EXPECT_THROW(NDArrayView::wrap(raw + 1, 15, {8}, BINARY), std::invalid_argument);
	EXPECT_THROW(NDArrayView::wrap(raw + 1, 15, {8}, INT3), std::invalid_argument);
}

TEST(NDArray_wrap, Mmap_ViewAndMutable) {
	const size_t page = (size_t)sysconf(_SC_PAGESIZE);
	void* p = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	ASSERT_NE(p, MAP_FAILED);
	float* f = static_cast<float*>(p);
	f[0] = 1.0f;
	f[1] = 2.0f;
	f[2] = 3.0f;
	{
		NDArrayView v = NDArrayView::wrap(p, 3 * sizeof(float), {3}, F32);
		EXPECT_FLOAT_EQ(v.getFlat<float>(2), 3.0f);
		NDArray w = NDArray::wrap(p, 3 * sizeof(float), {3}, F32);
		w.set({2}, 8.0f);
		EXPECT_FLOAT_EQ(f[2], 8.0f);
		EXPECT_FLOAT_EQ(v.getFlat<float>(2), 8.0f);
	}
	EXPECT_FLOAT_EQ(f[0], 1.0f);
	EXPECT_FLOAT_EQ(f[2], 8.0f);
	ASSERT_EQ(munmap(p, page), 0);
}

TEST(NDArray_wrap, ViewOutlivesWrapHandle) {
	float buf[2] = {10.0f, 20.0f};
	NDArrayView v;
	{
		NDArray a = NDArray::wrap(buf, sizeof(buf), {2}, F32);
		v = a.view();
	}
	EXPECT_FLOAT_EQ(v.getFlat<float>(0), 10.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(1), 20.0f);
	EXPECT_FALSE(v.sharedBuffer().get()->ownsData);
}

TEST(NDArray_wrap, RowStride_F32_GetFlatAndRow) {
	const int nRows = 3;
	const int dims = 10;
	const size_t stride = 64;
	alignas(4) uint8_t buf[3 * 64];
	memset(buf, 0, sizeof(buf));
	for (int i = 0; i < nRows; ++i) {
		NDArray row = NDArray::wrap(buf + (size_t)i * stride, (size_t)dims * sizeof(float),
		                            {dims}, F32);
		for (int j = 0; j < dims; ++j)
			row.set({j}, (float)(i * 100 + j));
	}

	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {nRows, dims}, F32, stride);
	EXPECT_FALSE(v.isContiguous());
	EXPECT_EQ(v.numElements(), (size_t)(nRows * dims));
	EXPECT_FLOAT_EQ(v.getFlat<float>(0), 0.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(9), 9.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(10), 100.0f);
	EXPECT_FLOAT_EQ(v.getFlat<float>(29), 209.0f);

	NDArrayView r1 = v.row(1);
	EXPECT_TRUE(r1.isContiguous());
	EXPECT_EQ(r1.numElements(), (size_t)dims);
	EXPECT_FLOAT_EQ(r1.getFlat<float>(0), 100.0f);
	EXPECT_FLOAT_EQ(r1.getFlat<float>(9), 109.0f);

	NDArray q({dims}, F32);
	for (int j = 0; j < dims; ++j)
		q.set({j}, 1.0f);
	float naive = 0;
	for (int j = 0; j < dims; ++j)
		naive += 100.0f + (float)j;
	EXPECT_NEAR(r1.dot<float>(q.view()), naive, 1e-4f);
}

TEST(NDArray_wrap, RowStride_LastRowNoTrailingPad) {
	const size_t stride = 64;
	const int dims = 10;
	const size_t need = stride + (size_t)dims * sizeof(float);
	alignas(4) uint8_t buf[64 + 40];
	memset(buf, 0, sizeof(buf));
	float* r0 = (float*)buf;
	float* r1 = (float*)(buf + stride);
	r0[0] = 1.0f;
	r1[9] = 2.0f;
	NDArrayView v = NDArrayView::wrap(buf, need, {2, dims}, F32, stride);
	EXPECT_FLOAT_EQ(v.row(0).getFlat<float>(0), 1.0f);
	EXPECT_FLOAT_EQ(v.row(1).getFlat<float>(9), 2.0f);
}

TEST(NDArray_wrap, RowStride_INT8) {
	const size_t stride = 64;
	alignas(1) uint8_t buf[2 * 64];
	memset(buf, 0, sizeof(buf));
	int8_t* r0 = (int8_t*)buf;
	int8_t* r1 = (int8_t*)(buf + stride);
	r0[0] = -5;
	r1[3] = 7;
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {2, 4}, INT8, stride);
	EXPECT_EQ(v.row(0).getFlat<int>(0), -5);
	EXPECT_EQ(v.row(1).getFlat<int>(3), 7);
}

TEST(NDArray_wrap, RowStride_INT3) {
	const size_t stride = 64;
	alignas(8) uint8_t buf[2 * 64];
	memset(buf, 0, sizeof(buf));
	NDArray a = NDArray::wrap(buf, 8, {16}, INT3);
	NDArray b = NDArray::wrap(buf + stride, 8, {16}, INT3);
	a.set({0}, -4);
	a.set({15}, 3);
	b.set({1}, -1);
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {2, 16}, INT3, stride);
	EXPECT_EQ(v.row(0).getFlat<int>(0), -4);
	EXPECT_EQ(v.row(0).getFlat<int>(15), 3);
	EXPECT_EQ(v.row(1).getFlat<int>(1), -1);
}

TEST(NDArray_wrap, RowStride_BINARY) {
	const size_t stride = 64;
	alignas(8) uint8_t buf[2 * 64];
	memset(buf, 0, sizeof(buf));
	NDArray a = NDArray::wrap(buf, 8, {16}, BINARY);
	NDArray b = NDArray::wrap(buf + stride, 8, {16}, BINARY);
	a.set({0}, 1);
	a.set({7}, 1);
	b.set({3}, 1);
	NDArrayView v = NDArrayView::wrap(buf, sizeof(buf), {2, 16}, BINARY, stride);
	EXPECT_EQ(v.row(0).getFlat<int>(0), 1);
	EXPECT_EQ(v.row(0).getFlat<int>(7), 1);
	EXPECT_EQ(v.row(1).getFlat<int>(3), 1);
}

TEST(NDArray_wrap, RowStride_UndersizedStrideThrows) {
	alignas(4) float buf[32] = {};
	EXPECT_THROW(NDArrayView::wrap(buf, sizeof(buf), {2, 10}, F32, 36), std::invalid_argument);
}

TEST(NDArray_wrap, RowStride_UnalignedStrideThrows) {
	alignas(4) uint8_t buf[128] = {};
	EXPECT_THROW(NDArrayView::wrap(buf, sizeof(buf), {2, 10}, F32, 63), std::invalid_argument);
}

TEST(NDArray_wrap, RowStride_ShortBufferThrows) {
	alignas(4) uint8_t buf[64 + 39] = {};
	EXPECT_THROW(NDArrayView::wrap(buf, sizeof(buf), {2, 10}, F32, 64), std::invalid_argument);
}

TEST(NDArray_wrap, RowStride_DenseArrayWrapOk) {
	float buf[6] = {1, 2, 3, 4, 5, 6};
	NDArray a = NDArray::wrap(buf, sizeof(buf), {2, 3}, F32, 12);
	EXPECT_FLOAT_EQ(a.get<float>({1, 2}), 6.0f);
}

TEST(NDArray_wrap, RowStride_PaddedArrayWrapThrows) {
	alignas(4) uint8_t buf[128] = {};
	EXPECT_THROW(NDArray::wrap(buf, sizeof(buf), {2, 10}, F32, 64), std::invalid_argument);
}

TEST(NDArray_wrap, OwnedCowStillDetaches) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b = a;
	EXPECT_TRUE(a.ownsStorage());
	a.set({0}, 99.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 99.0f);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 1.0f);
}

TEST(NDArray_wrap, OutOfPlaceUnary_DoesNotWriteThrough) {
	float buf[4] = {1.0f, -2.0f, 3.0f, -4.0f};
	const float orig[4] = {1.0f, -2.0f, 3.0f, -4.0f};
	NDArray a = NDArray::wrap(buf, sizeof(buf), {4}, F32);
	NDArray n = -a;
	NDArray sq = a.squared();
	NDArray ab = a.absolute();
	NDArray ng = a.negated();
	EXPECT_TRUE(n.ownsStorage());
	EXPECT_TRUE(sq.ownsStorage());
	EXPECT_TRUE(ab.ownsStorage());
	EXPECT_TRUE(ng.ownsStorage());
	EXPECT_FALSE(a.ownsStorage());
	EXPECT_NE(n.data(), static_cast<const void*>(buf));
	EXPECT_FLOAT_EQ(n.get<float>({0}), -1.0f);
	EXPECT_FLOAT_EQ(n.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(sq.get<float>({1}), 4.0f);
	EXPECT_FLOAT_EQ(ab.get<float>({1}), 2.0f);
	EXPECT_FLOAT_EQ(ng.get<float>({0}), -1.0f);
	for (int i = 0; i < 4; ++i)
		EXPECT_FLOAT_EQ(buf[i], orig[i]);
}

TEST(NDArray_wrap, SoftmaxedAndSilued_DoNotWriteThrough) {
	float buf[2] = {0.0f, 0.0f};
	NDArray a = NDArray::wrap(buf, sizeof(buf), {2}, F32);
	NDArray s = a.softmaxed();
	NDArray u = a.silued();
	EXPECT_TRUE(s.ownsStorage());
	EXPECT_TRUE(u.ownsStorage());
	EXPECT_FLOAT_EQ(buf[0], 0.0f);
	EXPECT_FLOAT_EQ(buf[1], 0.0f);
	EXPECT_NEAR(s.get<float>({0}), 0.5f, 1e-5f);
	EXPECT_NEAR(s.get<float>({1}), 0.5f, 1e-5f);
}

TEST(NDArray_wrap, LogicalNot_Binary_DoesNotWriteThrough) {
	alignas(8) uint64_t words[1] = {1ull};
	const uint64_t orig = words[0];
	NDArray a = NDArray::wrap(words, sizeof(words), {8}, BINARY);
	NDArray n = ~a;
	EXPECT_TRUE(n.ownsStorage());
	EXPECT_FALSE(a.ownsStorage());
	EXPECT_EQ(words[0], orig);
	EXPECT_EQ(a.get<int>({0}), 1);
	EXPECT_EQ(n.get<int>({0}), 0);
	EXPECT_EQ(n.get<int>({1}), 1);
}

TEST(NDArray_wrap, OwnedOutOfPlace_IsDistinctBuffer) {
	NDArray a(ArrayList({1.0f, 2.0f}));
	NDArray n = -a;
	EXPECT_TRUE(n.ownsStorage());
	EXPECT_TRUE(a.ownsStorage());
	EXPECT_NE(n.data(), a.data());
	EXPECT_FLOAT_EQ(a.get<float>({0}), 1.0f);
	EXPECT_FLOAT_EQ(n.get<float>({0}), -1.0f);
}

TEST(NDArray_wrap, PackedByteSize_StaticByCount) {
	EXPECT_EQ(NDArray::packedByteSize(F32, (size_t)0), (size_t)0);
	EXPECT_EQ(NDArray::packedByteSize(F32, (size_t)3), (size_t)12);
	EXPECT_EQ(NDArray::packedByteSize(F64, (size_t)2), (size_t)16);
	EXPECT_EQ(NDArray::packedByteSize(INT32, (size_t)4), (size_t)16);
	EXPECT_EQ(NDArray::packedByteSize(INT64, (size_t)1), (size_t)8);
	EXPECT_EQ(NDArray::packedByteSize(UINT8, (size_t)7), (size_t)7);
	EXPECT_EQ(NDArray::packedByteSize(INT8, (size_t)7), (size_t)7);
	EXPECT_EQ(NDArray::packedByteSize(F16, (size_t)5), (size_t)10);
	EXPECT_EQ(NDArray::packedByteSize(BF16, (size_t)5), (size_t)10);
	EXPECT_EQ(NDArray::packedByteSize(UINT256, (size_t)2), (size_t)64);
	EXPECT_EQ(NDArray::packedByteSize(BINARY, (size_t)0), (size_t)0);
	EXPECT_EQ(NDArray::packedByteSize(BINARY, (size_t)1), (size_t)8);
	EXPECT_EQ(NDArray::packedByteSize(BINARY, (size_t)64), (size_t)8);
	EXPECT_EQ(NDArray::packedByteSize(BINARY, (size_t)65), (size_t)16);
	EXPECT_EQ(NDArray::packedByteSize(INT3, (size_t)0), (size_t)0);
	EXPECT_EQ(NDArray::packedByteSize(INT3, (size_t)1), (size_t)8);
	EXPECT_EQ(NDArray::packedByteSize(INT3, (size_t)16), (size_t)8);
	EXPECT_EQ(NDArray::packedByteSize(INT3, (size_t)17), (size_t)16);
}

TEST(NDArray_wrap, PackedByteSize_StaticByShape) {
	EXPECT_EQ(NDArray::packedByteSize(F32, ArrayList<int>({2, 3})), (size_t)24);
	EXPECT_EQ(NDArray::packedByteSize(INT3, ArrayList<int>({2, 16})), (size_t)16);
	EXPECT_EQ(NDArray::packedByteSize(BINARY, ArrayList<int>({4, 16})), (size_t)8);
	EXPECT_EQ(NDArray::packedByteSize(F32, ArrayList<int>({0, 8})), (size_t)0);
	EXPECT_EQ(NDArray::packedByteSize(UINT8, ArrayList<int>()), (size_t)1);
	EXPECT_THROW(NDArray::packedByteSize(F32, ArrayList<int>({-1, 4})), std::invalid_argument);
	EXPECT_THROW(NDArray::packedByteSize(UINT256, ArrayList<int>({1 << 30, 1 << 30})),
	             std::invalid_argument);
}

TEST(NDArray_wrap, PackedByteSize_InstanceMatchesStaticAndOwnerBuffer) {
	NDArray a({3, 4}, F32);
	EXPECT_EQ(a.packedByteSize(), NDArray::packedByteSize(F32, (size_t)12));
	EXPECT_EQ(a.packedByteSize(), a.byteSize());
	EXPECT_EQ(a.view().packedByteSize(), a.packedByteSize());

	NDArray i3({17}, INT3);
	EXPECT_EQ(i3.packedByteSize(), (size_t)16);
	EXPECT_EQ(i3.packedByteSize(), i3.byteSize());
	EXPECT_EQ(i3.view().packedByteSize(), (size_t)16);
}

TEST(NDArray_wrap, PackedByteSize_WrapPadIsLargerBuffer) {
	alignas(4) float buf[8] = {};
	NDArray w = NDArray::wrap(buf, sizeof(buf), {4}, F32);
	EXPECT_EQ(w.packedByteSize(), (size_t)16);
	EXPECT_EQ(w.byteSize(), sizeof(buf));
	EXPECT_GT(w.byteSize(), w.packedByteSize());
	EXPECT_EQ(w.view().packedByteSize(), (size_t)16);
	EXPECT_EQ(w.view().byteSize(), sizeof(buf));
}

TEST(NDArray_wrap, PackedByteSize_SliceIsLogicalCount) {
	NDArray a({4, 4}, F32);
	NDArrayView row = a.row(1);
	EXPECT_EQ(row.packedByteSize(), NDArray::packedByteSize(F32, (size_t)4));
	EXPECT_EQ(row.byteSize(), a.byteSize());
}

TEST(NDArray_wrap, PackedByteSize_EmptyIsZero) {
	NDArray a = NDArray::empty(INT3);
	EXPECT_EQ(a.packedByteSize(), (size_t)0);
	EXPECT_EQ(a.view().packedByteSize(), (size_t)0);
	EXPECT_EQ(NDArray::packedByteSize(INT3, (size_t)0), (size_t)0);
}
