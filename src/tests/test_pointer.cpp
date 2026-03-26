//
// Created by komozoi on 5.3.26.
//

#include <gtest/gtest.h>
#include <alloc/pointer.h>


TEST(SpTest, DefaultIsNull) {
	sp<int> p;
	EXPECT_FALSE(p);
	EXPECT_EQ(p.get(), nullptr);
}

TEST(SpTest, ConstructFromRawPointer) {
	sp<int> p(new int(42));
	EXPECT_TRUE(p);
	EXPECT_EQ(*p, 42);
}

TEST(SpTest, MoveConstructorTransfersOwnership) {
	sp<int> a(new int(7));
	sp<int> b(std::move(a));

	EXPECT_FALSE(a);
	EXPECT_TRUE(b);
	EXPECT_EQ(*b, 7);
}

TEST(SpTest, MoveAssignmentTransfersOwnership) {
	sp<int> a(new int(5));
	sp<int> b;

	b = std::move(a);

	EXPECT_FALSE(a);
	EXPECT_TRUE(b);
	EXPECT_EQ(*b, 5);
}

TEST(SpTest, ReleaseTransfersRawPointer) {
	sp<int> p(new int(99));

	int* raw = p.release();

	EXPECT_FALSE(p);
	EXPECT_EQ(*raw, 99);

	delete raw;
}

TEST(SpTest, ResetDeletesObject) {
	static int destroyed = 0;
	struct Counter {
		~Counter() { destroyed++; }
	};

	{
		sp<Counter> p(new Counter());
		p.reset();
		EXPECT_EQ(destroyed, 1);
	}

	EXPECT_EQ(destroyed, 1);
}

TEST(SpTest, CopyFromUniqueCreatesCopyOnWrite) {
	sp<int> a(new int(10), SpPointerType::UNIQUE);

	sp<int> b = a;   // enable copy later

	EXPECT_TRUE(a);
	EXPECT_TRUE(b);
	EXPECT_EQ(*a, 10);
	EXPECT_EQ(*b, 10);

	*b = 20;  // should trigger deep copy

	EXPECT_EQ(*b, 20);
	EXPECT_EQ(*a, 10);
}

TEST(SpTest, MultipleCopyOnWriteIsolation) {
	sp<int> a(new int(1), SpPointerType::UNIQUE);
	sp<int> b = a;
	sp<int> c = a;

	*b = 2;
	*c = 3;

	EXPECT_EQ(*a, 1);
	EXPECT_EQ(*b, 2);
	EXPECT_EQ(*c, 3);
}

TEST(SpTest, SharedDoesNotCopyOnWrite) {
	sp<int> a(5, SpPointerType::SHARED);
	sp<int> b = a;

	*b = 8;

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
		~Counter() { destroyed++; }
	};

	{
		sp<Counter> a(Counter());
		sp<Counter> b = a;
		*b;  // no write, no COW yet
	}

	EXPECT_EQ(destroyed, 1);
}

TEST(SpTest, WriteTriggersSingleClone) {
	struct Data {
		int value;
	};

	sp<Data> a(new Data{1});
	sp<Data> b = a;

	EXPECT_EQ(a.get(), b.get());

	b->value = 2;

	EXPECT_NE(a.get(), b.get());
	EXPECT_EQ(a->value, 1);
	EXPECT_EQ(b->value, 2);
}


