//
// Created by nuclaer on 5.3.26.
//

#include <gtest/gtest.h>
#include <alloc/pointer.h>


TEST(SpTest, DefaultIsNull) {
	sp<int> p;
	EXPECT_FALSE(p);
	EXPECT_EQ(p.get(), nullptr);
}

TEST(SpTest, ConstructFromRawPointer) {
	sp<int> p(42);
	EXPECT_TRUE(p);
	EXPECT_EQ(*p, 42);
}

TEST(SpTest, MoveConstructorTransfersOwnership) {
	sp<int> a(7);
	sp<int> b(std::move(a));

	EXPECT_FALSE(a);
	EXPECT_TRUE(b);
	EXPECT_EQ(*b, 7);
}

TEST(SpTest, MoveAssignmentTransfersOwnership) {
	sp<int> a(5);
	sp<int> b;

	b = std::move(a);

	EXPECT_FALSE(a);
	EXPECT_TRUE(b);
	EXPECT_EQ(*b, 5);
}

TEST(SpTest, ResetDeletesObject) {
	static int destroyed = 0;
	struct Counter {
		~Counter() { destroyed++; }
	};

	{
		sp<Counter> p = sp<Counter>::create();
		p.reset();
		EXPECT_EQ(destroyed, 1);
	}

	EXPECT_EQ(destroyed, 1);
}

TEST(SpTest, CopyFromUniqueCreatesCopyOnWrite) {
	sp<int> a(SpPointerType::UNIQUE, 10);

	sp<int> b = a;

	EXPECT_TRUE(a);
	EXPECT_TRUE(b);
	EXPECT_EQ(*a, 10);
	EXPECT_EQ(*b, 10);

	b.mut() = 20;  // should trigger deep copy

	EXPECT_EQ(*b, 20);
	EXPECT_EQ(*a, 10);
}

TEST(SpTest, MultipleCopyOnWriteIsolation) {
	sp<int> a(SpPointerType::UNIQUE, 1);
	sp<int> b = a;
	sp<int> c = a;

	b.mut() = 2;
	c.mut() = 3;

	EXPECT_EQ(*a, 1);
	EXPECT_EQ(*b, 2);
	EXPECT_EQ(*c, 3);
}

TEST(SpTest, SharedDoesNotCopyOnWrite) {
	sp<int> a(SpPointerType::SHARED, 5);
	sp<int> b = a;

	b.mut() = 8;

	EXPECT_EQ(*a, 8);
	EXPECT_EQ(*b, 8);
}

TEST(SpTest, MovePreservesUnique) {
	sp<int> a(100);

	sp<int> b(std::move(a));

	EXPECT_TRUE(b);
	EXPECT_EQ(*b, 100);
}

TEST(SpTest, CopyOnWriteDeletesExactlyOnce) {
	static int destroyed = 0;
	struct Counter {
		Counter() = default;
		Counter(Counter&& moved)  noexcept {}
		~Counter() { destroyed++; }
		void doNothing() const {}
	};

	{
		sp<Counter> a = sp<Counter>::create();
		sp<Counter> b = a; // NOLINT(*-unnecessary-copy-initialization)
		b->doNothing();
	}

	EXPECT_EQ(destroyed, 1);
}

TEST(SpTest, WriteTriggersSingleClone) {
	struct Data {
		int value;
	};

	sp<Data> a(Data{1});
	sp<Data> b = a;

	EXPECT_EQ(a.get(), b.get());

	b.mut().value = 2;

	EXPECT_NE(a.get(), b.get());
	EXPECT_EQ(a->value, 1);
	EXPECT_EQ(b->value, 2);
}

TEST(SpTest, CopyFromUniqueCreatesCOW) {
	sp<int> a(SpPointerType::UNIQUE, 5);
	sp<int> b = a;

	EXPECT_EQ(b.pointerType(), SpPointerType::COPY_ON_WRITE);
	EXPECT_EQ(a.pointerType(), SpPointerType::UNIQUE);
}

TEST(SpTest, COWDetachesOnMut) {
	sp<int> a(SpPointerType::UNIQUE, 10);
	sp<int> b = a;

	b.mut() = 20;

	EXPECT_EQ(*a, 10);
	EXPECT_EQ(*b, 20);
	EXPECT_EQ(b.pointerType(), SpPointerType::SHARED);

	sp<int> c = b;

	c.mut() = 30;

	EXPECT_EQ(*b, 30);
	EXPECT_EQ(*c, 30);
	EXPECT_EQ(c.pointerType(), SpPointerType::SHARED);
}

TEST(SpTest, SharedDoesNotDetach) {
	sp<int> a(SpPointerType::SHARED, 1);
	sp<int> b = a;

	b.mut() = 5;

	EXPECT_EQ(*a, 5);
	EXPECT_EQ(*b, 5);
}

TEST(SpTest, DestroyedExactlyOncePerBlock) {
	static int destroyed = 0;

	struct Counter {
		~Counter() { destroyed++; }
	};

	{
		sp<Counter> a(SpPointerType::UNIQUE);
		sp<Counter> b = a;
	}

	EXPECT_EQ(destroyed, 1);
}
