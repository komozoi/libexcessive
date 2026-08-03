/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-08-03
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
#include <string>
#include <vector>


// ---------------------------------------------------------------------------
// Hierarchy fixtures: SpManaged is on the superclass; subclasses use ptr().
// ---------------------------------------------------------------------------

struct Entity : public SpManaged<Entity, SHARED> {
	int id = 0;
	mutable int visitCount = 0;

	explicit Entity(int id_ = 0) : id(id_) {}
	virtual ~Entity() = default;

	virtual std::string name() const {
		return "Entity";
	}

	/** Base API that accepts a smart pointer instead of a raw `this`. */
	static void touch(const sp<Entity>& e) {
		if (e)
			e->visitCount++;
	}

	static void touch(const sp<const Entity>& e) {
		if (e)
			e->visitCount++;
	}

	/** Uses ptr() rather than `this` when calling into pointer-based APIs. */
	void registerSelf() {
		touch(ptr());
	}

	void registerSelf() const {
		touch(ptr());
	}
};

struct Animal : public Entity {
	std::string species;

	Animal() : Entity(0), species("unknown") {}
	explicit Animal(int id_, std::string species_)
		: Entity(id_), species(std::move(species_)) {}

	std::string name() const override {
		return "Animal:" + species;
	}

	/**
	 * Subclass method: obtain sp<Entity> via ptr() and use it instead of raw this.
	 * Returns the registered type (superclass), not sp<Animal>.
	 */
	sp<Entity> asEntityPtr() {
		return ptr();
	}

	sp<const Entity> asEntityPtr() const {
		return ptr();
	}

	/** Pass self into a helper that only knows about Entity. */
	void announce(std::vector<sp<Entity>>& out) {
		out.push_back(ptr());
	}
};

struct Dog : public Animal {
	std::string breed;

	Dog() : Animal(0, "dog"), breed("mutt") {}
	Dog(int id_, std::string breed_)
		: Animal(id_, "dog"), breed(std::move(breed_)) {}

	std::string name() const override {
		return "Dog:" + breed;
	}

	/** Deep subclass still uses inherited ptr() → sp<Entity>. */
	sp<Entity> identity() {
		return ptr();
	}

	void barkInto(std::vector<std::string>& log) {
		sp<Entity> self = ptr();
		log.push_back(self->name() + "#" + std::to_string(self->id));
	}
};

struct Cat : public Animal {
	bool indoor = true;

	Cat() : Animal(0, "cat") {}
	explicit Cat(int id_, bool indoor_ = true)
		: Animal(id_, "cat"), indoor(indoor_) {}

	std::string name() const override {
		return indoor ? "Cat:indoor" : "Cat:outdoor";
	}

	sp<Entity> identity() {
		return ptr();
	}
};

// Intermediate layer with no extra SpManaged (still under Entity's SpManaged).
struct Mammal : public Animal {
	explicit Mammal(int id_ = 0) : Animal(id_, "mammal") {}

	std::string name() const override {
		return "Mammal";
	}

	sp<Entity> selfEntity() {
		return ptr();
	}
};

struct Whale : public Mammal {
	explicit Whale(int id_ = 0) : Mammal(id_) {
		species = "whale";
	}

	std::string name() const override {
		return "Whale";
	}

	/** Multi-level: Entity <- Animal <- Mammal <- Whale */
	void surface(std::vector<sp<Entity>>& school) {
		school.push_back(ptr());
	}
};


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(SpManagedTest, BasicPtrReturnsSameInstance) {
	sp<Entity> e = sp<Entity>::create(7);
	sp<Entity> self = e.mut().ptr();

	EXPECT_TRUE(self);
	EXPECT_EQ(self.get(), e.get());
	EXPECT_EQ(self->id, 7);
	EXPECT_EQ(e.numReferences(), 2);
}

TEST(SpManagedTest, UnownedObjectThrows) {
	Entity stack(1);
	EXPECT_THROW(stack.ptr(), std::bad_weak_ptr);
}

TEST(SpManagedTest, DerivedPtrReturnsSpOfSuperclass) {
	sp<Animal> a = sp<Animal>::create(3, "fox");

	// ptr() is inherited from SpManaged<Entity> → sp<Entity>
	sp<Entity> asBase = a.mut().ptr();
	EXPECT_TRUE(asBase);
	EXPECT_EQ(asBase.get(), static_cast<Entity*>(a.get()));
	EXPECT_EQ(asBase->id, 3);
	EXPECT_EQ(asBase->name(), "Animal:fox");

	// Same via subclass helper
	sp<Entity> viaHelper = a.mut().asEntityPtr();
	EXPECT_EQ(viaHelper.get(), asBase.get());
	EXPECT_EQ(a.numReferences(), 3);
}

TEST(SpManagedTest, DerivedUsesPtrInsteadOfThis) {
	sp<Dog> dog = sp<Dog>::create(10, "corgi");

	// registerSelf() calls touch(ptr()) — no raw this handed to the API
	dog.mut().registerSelf();
	EXPECT_EQ(dog->visitCount, 1);

	std::vector<std::string> log;
	dog.mut().barkInto(log);
	ASSERT_EQ(log.size(), 1u);
	EXPECT_EQ(log[0], "Dog:corgi#10");

	// identity() returns sp<Entity> to the same object
	sp<Entity> id = dog.mut().identity();
	EXPECT_EQ(id.get(), static_cast<Entity*>(dog.get()));
	EXPECT_EQ(id->name(), "Dog:corgi");
}

TEST(SpManagedTest, MultiLevelHierarchyPtr) {
	sp<Whale> w = sp<Whale>::create(99);

	sp<Entity> e1 = w.mut().ptr();
	sp<Entity> e2 = w.mut().selfEntity();

	EXPECT_EQ(e1.get(), e2.get());
	EXPECT_EQ(e1.get(), static_cast<Entity*>(w.get()));
	EXPECT_EQ(e1->name(), "Whale");
	EXPECT_EQ(e1->id, 99);

	std::vector<sp<Entity>> school;
	w.mut().surface(school);
	ASSERT_EQ(school.size(), 1u);
	EXPECT_EQ(school[0].get(), e1.get());
	// owner + e1 + e2 + school[0]
	EXPECT_GE(w.numReferences(), 4);
}

TEST(SpManagedTest, SiblingDerivedTypesShareBasePtrType) {
	sp<Dog> dog = sp<Dog>::create(1, "lab");
	sp<Cat> cat = sp<Cat>::create(2, true);

	std::vector<sp<Entity>> zoo;
	dog.mut().announce(zoo);
	// Cat inherits Animal::announce
	cat.mut().announce(zoo);

	ASSERT_EQ(zoo.size(), 2u);
	EXPECT_EQ(zoo[0]->name(), "Dog:lab");
	EXPECT_EQ(zoo[1]->name(), "Cat:indoor");
	EXPECT_NE(zoo[0].get(), zoo[1].get());

	// Both entries are sp<Entity>, not the derived sp types
	static_assert(std::is_same_v<decltype(zoo[0]), sp<Entity>&>, "vector holds sp<Entity>");
}

TEST(SpManagedTest, ConvertDerivedSpToBaseSpKeepsPtrWorking) {
	sp<Dog> dog = sp<Dog>::create(5, "beagle");
	sp<Entity> asEntity = dog; // templated conversion

	EXPECT_EQ(asEntity.get(), static_cast<Entity*>(dog.get()));
	EXPECT_EQ(dog.numReferences(), 2);

	// ptr() from the derived object still yields the same control block
	sp<Entity> fromPtr = dog.mut().ptr();
	EXPECT_EQ(fromPtr.get(), asEntity.get());
	EXPECT_EQ(dog.numReferences(), 3);

	// Calling through base view: Entity::registerSelf uses ptr()
	asEntity.mut().registerSelf();
	EXPECT_EQ(dog->visitCount, 1);
}

TEST(SpManagedTest, ConstPtrFromConstMethod) {
	sp<Animal> a = sp<Animal>::create(4, "owl");

	const Animal& cref = *a.get();
	sp<const Entity> cself = cref.asEntityPtr();
	EXPECT_TRUE(cself);
	EXPECT_EQ(cself.get(), static_cast<const Entity*>(a.get()));
	EXPECT_EQ(cself->name(), "Animal:owl");

	// const registerSelf → const ptr()
	cref.registerSelf();
	EXPECT_EQ(a->visitCount, 1);
}

TEST(SpManagedTest, PolymorphicDispatchThroughPtr) {
	sp<Dog> dog = sp<Dog>::create(8, "poodle");
	sp<Entity> e = dog.mut().ptr();

	// Virtual call through sp<Entity> obtained from subclass ptr()
	EXPECT_EQ(e->name(), "Dog:poodle");

	sp<Entity> animalView = sp<Animal>(dog).mut().ptr();
	EXPECT_EQ(animalView.get(), e.get());
	EXPECT_EQ(animalView->name(), "Dog:poodle");
}

TEST(SpManagedTest, MultiplePtrCallsShareOwnership) {
	sp<Cat> cat = sp<Cat>::create(11, false);
	EXPECT_EQ(cat.numReferences(), 1);

	sp<Entity> p1 = cat.mut().identity();
	sp<Entity> p2 = cat.mut().ptr();
	sp<Entity> p3 = cat.mut().asEntityPtr();

	EXPECT_EQ(p1.get(), p2.get());
	EXPECT_EQ(p2.get(), p3.get());
	EXPECT_EQ(cat.numReferences(), 4);

	p1.reset();
	p2.reset();
	EXPECT_EQ(cat.numReferences(), 2);
	EXPECT_EQ(cat->name(), "Cat:outdoor");
}

TEST(SpManagedTest, PtrAfterMoveOfOwningSp) {
	sp<Dog> dog = sp<Dog>::create(12, "husky");
	sp<Dog> moved = std::move(dog);
	EXPECT_FALSE(dog);
	EXPECT_TRUE(moved);

	sp<Entity> self = moved.mut().ptr();
	EXPECT_EQ(self.get(), static_cast<Entity*>(moved.get()));
	EXPECT_EQ(self->id, 12);
}

TEST(SpManagedTest, CreateBaseTypeDirectly) {
	sp<Entity> e(SpPointerType::SHARED, 42);
	sp<Entity> self = e.mut().ptr();
	EXPECT_EQ(self.get(), e.get());
	EXPECT_EQ(self->name(), "Entity");
}

TEST(SpManagedTest, SubclassHelperChainUsesOnlySmartPointers) {
	// Simulates internal code that never stores raw this for lifetime/ownership.
	sp<Whale> w = sp<Whale>::create(100);

	auto process = [](sp<Entity> e) {
		EXPECT_TRUE(e);
		e.mut().registerSelf();
		return e->name();
	};

	std::string n = process(w.mut().ptr());
	EXPECT_EQ(n, "Whale");
	EXPECT_EQ(w->visitCount, 1);
}
