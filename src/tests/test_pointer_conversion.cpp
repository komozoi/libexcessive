/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-4-6
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
 *
 */

#include <gtest/gtest.h>
#include <alloc/pointer.h>
#include <string>
#include <utility>


class Base {
public:
    virtual int mutate() = 0;
    virtual int getValue() const = 0;

    virtual ~Base() = default;
};

class Derived : public Base {
public:
    int mutate() override { return 1; }
    int getValue() const override { return 2; }
};


TEST(SpConversionTest, derivedToBaseMove) {
    sp<Derived> d = sp<Derived>::create();
    EXPECT_TRUE(d);
    const sp<Base> b = std::move(d);
    EXPECT_TRUE(b);
    EXPECT_FALSE(d);
    EXPECT_EQ(b->getValue(), 2);
}

TEST(SpConversionTest, derivedToBaseCopy) {
    sp<Derived> d = sp<Derived>::create();
    EXPECT_TRUE(d);
    const sp<Base> b = d;
    EXPECT_TRUE(b);
    EXPECT_EQ(b->getValue(), 2);
    EXPECT_EQ(d->getValue(), 2);
}

TEST(SpConversionTest, derivedToBaseMoveWithMutation) {
    sp<Derived> d = sp<Derived>::create();
    EXPECT_TRUE(d);
    sp<Base> b = std::move(d);
    EXPECT_TRUE(b);
    EXPECT_FALSE(d);
    EXPECT_EQ(b->getValue(), 2);
    EXPECT_EQ(b.mut().mutate(), 1);
}

TEST(SpConversionTest, derivedToBaseCopyWithMutation) {
    sp<Derived> d = sp<Derived>::create();
    EXPECT_TRUE(d);
    sp<Base> b = d;
    EXPECT_TRUE(b);
    EXPECT_EQ(b->getValue(), 2);
    EXPECT_EQ(d->getValue(), 2);
    EXPECT_EQ(b.mut().mutate(), 1);
}

TEST(SpConversionTest, derivedToBaseAssignmentMove) {
    sp<Derived> d = sp<Derived>::create();
    sp<Base> b;
    EXPECT_FALSE(b);
    b = std::move(d);
    EXPECT_TRUE(b);
    EXPECT_FALSE(d);
    EXPECT_EQ(b->getValue(), 2);
}

TEST(SpConversionTest, derivedToBaseAssignmentCopy) {
    const sp<Derived> d = sp<Derived>::create();
    sp<Base> b;
    EXPECT_FALSE(b);
    b = d;
    EXPECT_TRUE(b);
    EXPECT_TRUE(d);
    EXPECT_EQ(b->getValue(), 2);
    EXPECT_EQ(d->getValue(), 2);
}

// ---------------------------------------------------------------------------
// Downcasts: Super -> Abstract -> Concrete (sp<Super> to sp<Abstract>)
// ---------------------------------------------------------------------------

class Super {
public:
	int tag = 0;
	explicit Super(int t = 0) : tag(t) {}
	virtual ~Super() = default;
	virtual const char* kind() const { return "Super"; }
};

class AbstractMid : public Super {
public:
	explicit AbstractMid(int t = 0) : Super(t) {}
	virtual int compute() const = 0;
	const char* kind() const override { return "AbstractMid"; }
};

class ConcreteLeaf : public AbstractMid {
public:
	explicit ConcreteLeaf(int t = 0) : AbstractMid(t) {}
	int compute() const override { return tag * 2; }
	const char* kind() const override { return "ConcreteLeaf"; }
};

/** Another Super that is not an AbstractMid — downcast must fail. */
class OtherSuper : public Super {
public:
	explicit OtherSuper(int t = 0) : Super(t) {}
	const char* kind() const override { return "OtherSuper"; }
};

TEST(SpConversionTest, superToAbstractConstructFromConcrete) {
	sp<ConcreteLeaf> leaf = sp<ConcreteLeaf>::create(21);
	sp<Super> asSuper = leaf;

	// Direct construction: sp<Abstract>(sp<Super>)
	sp<AbstractMid> mid(asSuper);
	EXPECT_TRUE(mid);
	EXPECT_EQ(mid.get(), static_cast<AbstractMid*>(leaf.get()));
	EXPECT_EQ(mid->compute(), 42);
	EXPECT_EQ(mid->kind(), std::string("ConcreteLeaf"));
	EXPECT_EQ(asSuper.numReferences(), 3); // leaf + asSuper + mid
}

TEST(SpConversionTest, superToAbstractCStyleCast) {
	sp<ConcreteLeaf> leaf = sp<ConcreteLeaf>::create(5);
	sp<Super> asSuper = leaf;

	// C-style cast: (sp<Abstract>)sp<Super>
	sp<AbstractMid> mid = (sp<AbstractMid>)asSuper;
	EXPECT_TRUE(mid);
	EXPECT_EQ(mid.get(), static_cast<AbstractMid*>(leaf.get()));
	EXPECT_EQ(mid->compute(), 10);
}

TEST(SpConversionTest, superToAbstractCopyAssignment) {
	sp<ConcreteLeaf> leaf = sp<ConcreteLeaf>::create(3);
	sp<Super> asSuper = leaf;

	sp<AbstractMid> mid;
	EXPECT_FALSE(mid);
	mid = asSuper;
	EXPECT_TRUE(mid);
	EXPECT_EQ(mid->compute(), 6);
	EXPECT_TRUE(asSuper);
}

TEST(SpConversionTest, superToAbstractMoveConstruct) {
	sp<ConcreteLeaf> leaf = sp<ConcreteLeaf>::create(7);
	sp<Super> asSuper = leaf;
	int refsBefore = asSuper.numReferences();

	sp<AbstractMid> mid(std::move(asSuper));
	EXPECT_TRUE(mid);
	EXPECT_FALSE(asSuper);
	EXPECT_EQ(mid->compute(), 14);
	EXPECT_EQ(mid.numReferences(), refsBefore); // ownership transferred, leaf still holds a ref
	EXPECT_TRUE(leaf);
}

TEST(SpConversionTest, superToAbstractMoveAssignment) {
	sp<ConcreteLeaf> leaf = sp<ConcreteLeaf>::create(9);
	sp<Super> asSuper = leaf;

	sp<AbstractMid> mid;
	mid = std::move(asSuper);
	EXPECT_TRUE(mid);
	EXPECT_FALSE(asSuper);
	EXPECT_EQ(mid->compute(), 18);
}

TEST(SpConversionTest, superToAbstractFailedCastYieldsNull) {
	sp<OtherSuper> other = sp<OtherSuper>::create(1);
	sp<Super> asSuper = other;

	sp<AbstractMid> mid(asSuper);
	EXPECT_FALSE(mid);

	sp<AbstractMid> midCast = (sp<AbstractMid>)asSuper;
	EXPECT_FALSE(midCast);

	// Source still valid after failed copy-cast
	EXPECT_TRUE(asSuper);
	EXPECT_EQ(asSuper->kind(), std::string("OtherSuper"));
}

TEST(SpConversionTest, superToAbstractFailedMoveLeavesSource) {
	sp<OtherSuper> other = sp<OtherSuper>::create(2);
	sp<Super> asSuper = other;

	sp<AbstractMid> mid(std::move(asSuper));
	EXPECT_FALSE(mid);
	// Move-downcast failed: source must not be stolen
	EXPECT_TRUE(asSuper);
	EXPECT_EQ(asSuper->tag, 2);
}

TEST(SpConversionTest, superToAbstractFailedMoveAssignLeavesBoth) {
	sp<OtherSuper> other = sp<OtherSuper>::create(2);
	sp<Super> asSuper = other;
	sp<AbstractMid> mid = sp<ConcreteLeaf>::create(1);

	mid = std::move(asSuper);
	EXPECT_TRUE(mid);
	EXPECT_EQ(mid->compute(), 2);
	EXPECT_TRUE(asSuper);
	EXPECT_EQ(asSuper->tag, 2);
}

TEST(SpConversionTest, abstractToSuperStillUpcasts) {
	sp<ConcreteLeaf> leaf = sp<ConcreteLeaf>::create(4);
	sp<AbstractMid> mid = leaf;
	sp<Super> up = mid;
	EXPECT_TRUE(up);
	EXPECT_EQ(up.get(), static_cast<Super*>(leaf.get()));
	EXPECT_EQ(up->tag, 4);
}

TEST(SpConversionTest, concreteToAbstractDirect) {
	sp<ConcreteLeaf> leaf = sp<ConcreteLeaf>::create(11);
	// Upcast Concrete -> Abstract (implicit convertible)
	sp<AbstractMid> mid = leaf;
	EXPECT_TRUE(mid);
	EXPECT_EQ(mid->compute(), 22);

	// Round-trip Super -> Abstract via construction and C-style cast
	sp<Super> sup = mid;
	sp<AbstractMid> back(sup);
	sp<AbstractMid> backCast = (sp<AbstractMid>)sup;
	EXPECT_EQ(back.get(), mid.get());
	EXPECT_EQ(backCast.get(), mid.get());
}
