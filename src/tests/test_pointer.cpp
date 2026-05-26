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
		// Initial state: Start with a UNIQUE pointer.
		sp<Tracker> a(SpPointerType::UNIQUE);

		// Copy UNIQUE to another instance; results in Copy-On-Write (COW) semantics.
		sp<Tracker> b = a;

		// Create another COW reference to the same memory block.
		sp<Tracker> c = a.getWritableCopy();

		// Inner scope to test alias management and mutation-triggered detaches.
		{
			sp<Tracker> d = c;                    // Reference count increases on original block.

			// Mutate COW handle to trigger detachment into a new SHARED block.
			b.mut();

			// Mutate second handle, detaching it from the original.
			c.mut();

			// Mutate final alias to ensure the original block is properly handled.
			d.mut();

			// All handles now manage independent or newly SHARED blocks.

			// Transfer ownership of a SHARED pointer.
			sp<Tracker> e = std::move(c);

			// Reference a SHARED block.
			sp<Tracker> f = e;

			// Perform an in-place mutation on a SHARED block.
			f.mut();

			// Reset a SHARED handle.
			e.reset();

			// Create a new independent chain.
			sp<Tracker> g(SpPointerType::UNIQUE);
			sp<Tracker> h = g;
			h.mut();                              // Detach occurs.

			// Cross-block assignment: release old reference and acquire new.
			b = h;
		}  // Scope exit: cleanup of local handles and their associated blocks.

		// Final destruction of top-level pointers.
	}  // Automatic cleanup of remaining blocks.

	EXPECT_EQ(constructed, destroyed);
}

struct Collapsible {
	bool collapsed;
	Collapsible() : collapsed(false) {}
	explicit Collapsible(const sp<Collapsible>&) : collapsed(true) {}
};

struct CollapsibleBase {
	bool collapsed;
	CollapsibleBase() : collapsed(false) {}
	explicit CollapsibleBase(const sp<Collapsible>&) : collapsed(true) {}
	virtual ~CollapsibleBase() = default;
};

struct CollapsibleDerived : public CollapsibleBase {
	CollapsibleDerived() = default;
};

TEST(SpTest, DoesNotCollapseOnCopy) {
	sp<Collapsible> a = sp<Collapsible>::create();
	sp<Collapsible> b(a);

	EXPECT_FALSE(b->collapsed) << "Constructor sp(U&&) was incorrectly used instead of copy constructor";
}

TEST(SpTest, DoesNotCollapseOnMove) {
	sp<Collapsible> a = sp<Collapsible>::create();
	sp<Collapsible> b(std::move(a));

	EXPECT_FALSE(b->collapsed) << "Constructor sp(U&&) was incorrectly used instead of move constructor";
}

TEST(SpTest, DoesNotCollapseOnTemplatedCopy) {
	sp<CollapsibleDerived> d = sp<CollapsibleDerived>::create();
	sp<CollapsibleBase> b(d);

	EXPECT_FALSE(b->collapsed) << "Constructor sp(U&&) was incorrectly used instead of templated copy constructor";
}

TEST(SpTest, DoesNotCollapseOnTemplatedMove) {
	sp<CollapsibleDerived> d = sp<CollapsibleDerived>::create();
	sp<CollapsibleBase> b(std::move(d));

	EXPECT_FALSE(b->collapsed) << "Constructor sp(U&&) was incorrectly used instead of templated move constructor";
}

TEST(SpTest, NumReferences) {
	sp<int> a;
	EXPECT_EQ(a.numReferences(), 0);

	sp<int> b(10);
	EXPECT_EQ(b.numReferences(), 1);

	sp<int> c = b;
	EXPECT_EQ(b.numReferences(), 2);
	EXPECT_EQ(c.numReferences(), 2);

	{
		sp<int> d = c;
		EXPECT_EQ(b.numReferences(), 3);
		EXPECT_EQ(c.numReferences(), 3);
		EXPECT_EQ(d.numReferences(), 3);
	}

	EXPECT_EQ(b.numReferences(), 2);
	EXPECT_EQ(c.numReferences(), 2);

	b.reset();
	EXPECT_EQ(b.numReferences(), 0);
	EXPECT_EQ(c.numReferences(), 1);
}

TEST(SpTest, NumReferencesWithCOW) {
	sp<int> a(SpPointerType::UNIQUE, 5);
	EXPECT_EQ(a.numReferences(), 1);

	sp<int> b = a;
	EXPECT_EQ(a.numReferences(), 2);
	EXPECT_EQ(b.numReferences(), 2);

	b.mut() = 10; // Should detach
	EXPECT_EQ(a.numReferences(), 1);
	EXPECT_EQ(b.numReferences(), 1);
}

TEST(SpTest, NumReferencesWithMove) {
	sp<int> a(5);
	EXPECT_EQ(a.numReferences(), 1);

	sp<int> b(std::move(a));
	EXPECT_EQ(a.numReferences(), 0);
	EXPECT_EQ(b.numReferences(), 1);
}

TEST(SpTest, NumReferencesWithTemplatedCopyMove) {
	struct Base { virtual ~Base() {} };
	struct Derived : Base {};

	sp<Derived> d = sp<Derived>::create();
	EXPECT_EQ(d.numReferences(), 1);

	sp<Base> b(d);
	EXPECT_EQ(d.numReferences(), 2);
	EXPECT_EQ(b.numReferences(), 2);

	sp<Base> b2(std::move(d));
	EXPECT_EQ(d.numReferences(), 0);
	EXPECT_EQ(b.numReferences(), 2);
	EXPECT_EQ(b2.numReferences(), 2);
}
