
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

TEST(NDArray_wrap, OwnedCowStillDetaches) {
	NDArray a(ArrayList({1.0f, 2.0f, 3.0f}));
	NDArray b = a;
	EXPECT_TRUE(a.ownsStorage());
	a.set({0}, 99.0f);
	EXPECT_FLOAT_EQ(a.get<float>({0}), 99.0f);
	EXPECT_FLOAT_EQ(b.get<float>({0}), 1.0f);
}
