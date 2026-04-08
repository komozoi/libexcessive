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
