/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-03-05
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


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

TEST(SpTest, StressTestComplexReferenceCounting) {
	static int constructed = 0;
	static int destroyed = 0;

	struct Tracker {
		Tracker() { ++constructed; }
		Tracker(const Tracker&) { ++constructed; }
		~Tracker() { ++destroyed; }
	};

	constructed = 0;
	destroyed = 0;

	{
		// Block A - start as UNIQUE
		sp<Tracker> a(SpPointerType::UNIQUE);

		// Copy UNIQUE, becomes COW (per current acquire_from_copy)
		sp<Tracker> b = a;

		// getWritableCopy(), another COW to same block
		sp<Tracker> c = a.getWritableCopy();

		// Inner scope with more aliases + mutations
		{
			sp<Tracker> d = c;                    // copy COW, another COW, refs=4 on block A

			// Trigger COW detach on b (COW + shared)
			b.mut();                              //, new block B, b becomes SHARED

			// Trigger COW detach on c
			c.mut();                              //, new block C, c becomes SHARED

			// Trigger COW detach on d (still pointing at old block A)
			d.mut();                              //, new block D, d becomes SHARED

			// Now we have 4 independent blocks (A,B,C,D)

			// Move a SHARED pointer
			sp<Tracker> e = std::move(c);         // e takes block C, c becomes null

			// Copy a SHARED pointer
			sp<Tracker> f = e;                    // refs on block C now 2

			// Mutate SHARED (no detach, just write)
			f.mut();

			// Reset while shared
			e.reset();                            // drops one ref on block C

			// New independent UNIQUE + copy + mutate
			sp<Tracker> g(SpPointerType::UNIQUE);
			sp<Tracker> h = g;
			h.mut();                              // triggers detach, block E for h (SHARED)

			// Cross-block assignment (releases old, acquires new)
			b = h;                                // b releases block B, takes block E (now SHARED)
		}  // d, e, f, g, h destruct here, should clean up blocks B, C (now 1), D, E

		// Final releases of remaining pointers
	}  // a, b destruct, block A and remaining block E

	EXPECT_EQ(constructed, destroyed);
}
