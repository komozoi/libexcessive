/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2023-11-02
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


#ifndef EXCESSIVE_HASHSET_H
#define EXCESSIVE_HASHSET_H


#include "stdio.h"
#include "alloc/StaticAllocator.h"
#include "alloc/SlabAllocator.h"
#include "ArrayList.h"
#include "Set.h"


template <class K>
class HashSet;

template <class K>
class HashSetIterator {
public:
	using iterator_category = std::bidirectional_iterator_tag;
	using value_type = K;
	using difference_type = std::ptrdiff_t;
	using pointer = K*;
	using reference = K&;

	HashSetIterator(HashSet<K>* set, int index) : set(set), index(index) {}
	K& operator*() { return set->keyAtIndex(index); }
	HashSetIterator& operator++() {
		index++;
		while (index < (int)set->getCapacity() && !set->presentAtIndex(index))
			index++;
		if (index >= (int)set->getCapacity())
			index = -1;
		return *this;
	}

	HashSetIterator operator++(int) {
		HashSetIterator tmp = *this;
		++(*this);
		return tmp;
	}

	HashSetIterator& operator--() {
		if (index == -1) {
			// Find last element
			for (int i = (int)set->getCapacity() - 1; i >= 0; i--) {
				if (set->presentAtIndex(i)) {
					index = i;
					return *this;
				}
			}
		} else {
			index--;
			while (index >= 0 && !set->presentAtIndex(index))
				index--;
			if (index < 0)
				index = -1; // Or should it be before-begin? STL reverse_iterator needs to reach one before begin.
		}
		return *this;
	}

	HashSetIterator operator--(int) {
		HashSetIterator tmp = *this;
		--(*this);
		return tmp;
	}

	bool operator==(const HashSetIterator& other) const { return set == other.set && index == other.index; }
	bool operator!=(const HashSetIterator& other) const { return !(*this == other); }
private:
	HashSet<K>* set;
	int index;
};

template <class K>
class HashSetConstIterator {
public:
	using iterator_category = std::bidirectional_iterator_tag;
	using value_type = const K;
	using difference_type = std::ptrdiff_t;
	using pointer = const K*;
	using reference = const K&;

	HashSetConstIterator(const HashSet<K>* set, int index) : set(set), index(index) {}
	const K& operator*() const { return set->keyAtIndex(index); }
	HashSetConstIterator& operator++() {
		index++;
		while (index < (int)set->getCapacity() && !set->presentAtIndex(index))
			index++;
		if (index >= (int)set->getCapacity())
			index = -1;
		return *this;
	}

	HashSetConstIterator operator++(int) {
		HashSetConstIterator tmp = *this;
		++(*this);
		return tmp;
	}

	HashSetConstIterator& operator--() {
		if (index == -1) {
			for (int i = (int)set->getCapacity() - 1; i >= 0; i--) {
				if (set->presentAtIndex(i)) {
					index = i;
					return *this;
				}
			}
		} else {
			index--;
			while (index >= 0 && !set->presentAtIndex(index))
				index--;
			if (index < 0)
				index = -1;
		}
		return *this;
	}

	HashSetConstIterator operator--(int) {
		HashSetConstIterator tmp = *this;
		--(*this);
		return tmp;
	}

	bool operator==(const HashSetConstIterator& other) const { return set == other.set && index == other.index; }
	bool operator!=(const HashSetConstIterator& other) const { return !(*this == other); }
private:
	const HashSet<K>* set;
	int index;
};

/**
 * @brief A hash-based implementation of the Set interface.
 *
 * This class uses open addressing and a bitmask for presence tracking.
 * It provides O(1) average time complexity for basic operations.
 *
 * @tparam K The type of elements in the set.
 */
template <class K>
class HashSet: public Set<K, HashSetIterator<K>, HashSetConstIterator<K>> {
	/**
	 * @brief Allocates memory for keys and the presence mask.
	 * @param cap The capacity to allocate for.
	 */
	void allocate(unsigned int cap) {
		unsigned int maskSize = (cap + 63) / 64;
		size_t totalSize = cap * sizeof(K) + maskSize * sizeof(uint64_t);
		keys = (K*)malloc(totalSize);
		mask = (uint64_t*)(keys + cap);
		for (unsigned int i = 0; i < maskSize; i++)
			mask[i] = 0;
	}

	/**
	 * @brief Checks if an element is present at the specified internal index.
	 * @param index The internal index.
	 * @return true if present, false otherwise.
	 */
	bool isPresent(unsigned int index) const {
		return (mask[index >> 6] >> (index & 0x3F)) & 1;
	}

	/**
	 * @brief Sets the presence flag for the specified internal index.
	 * @param index The internal index.
	 * @param present The presence flag value.
	 */
	void setPresent(unsigned int index, bool present) {
		if (present)
			mask[index >> 6] |= (1ULL << (index & 0x3F));
		else
			mask[index >> 6] &= ~(1ULL << (index & 0x3F));
	}

public:
	typedef HashSetIterator<K> Iterator;
	typedef HashSetConstIterator<K> ConstIterator;

	Iterator begin() override {
		for (unsigned int i = 0; i < capacity; i++)
			if (isPresent(i)) return Iterator(this, i);
		return end();
	}
	Iterator end() override { return Iterator(this, -1); }

	ConstIterator begin() const override {
		for (unsigned int i = 0; i < capacity; i++)
			if (isPresent(i)) return ConstIterator(this, i);
		return end();
	}
	ConstIterator end() const override { return ConstIterator(this, -1); }

	int size() const override { return amountUsed; }

	/**
	 * @brief Constructs a HashSet with the specified initial capacity.
	 * @param capacity Initial capacity.
	 */
	explicit HashSet(unsigned int capacity) : capacity(capacity) {
		if (capacity == 0)
			capacity = 1;
		allocate(capacity);
	}

	/**
	 * @brief Constructs a HashSet from a raw array.
	 * @param src Pointer to the source array.
	 * @param qty Number of elements to copy.
	 */
	explicit HashSet(const K* src, unsigned int qty) : capacity((qty * 7) / 5 + 1) {
		if (capacity == 0)
			capacity = 1;
		allocate(capacity);

		for (unsigned int i = 0; i < qty; i++)
			add(src[i]);
	}

	/**
	 * @brief Copy constructor.
	 * @param other The HashSet to copy from.
	 */
	HashSet(const HashSet<K>& other)
			: capacity(other.capacity), amountUsed(0) {
		allocate(capacity);
		addFrom(other);
	}

	/**
	 * @brief Move constructor.
	 * @param other The HashSet to move from.
	 */
	HashSet(HashSet<K>&& other) noexcept
			: capacity(other.capacity), amountUsed(other.amountUsed), keys(other.keys), mask(other.mask) {
		other.amountUsed = 0;
		other.capacity = 0;
		other.keys = nullptr;
		other.mask = nullptr;
	}

	/**
	 * @brief Copy assignment operator.
	 * @param other The HashSet to copy from.
	 * @return Reference to this HashSet.
	 */
	HashSet& operator=(const HashSet<K>& other) {
		if (&other == this)
			return *this;

		clear();
		if (keys)
			free(keys);

		capacity = other.capacity;
		allocate(capacity);
		addFrom(other);

		return *this;
	}

	/**
	 * @brief Move assignment operator.
	 * @param other The HashSet to move from.
	 * @return Reference to this HashSet.
	 */
	HashSet& operator=(HashSet<K>&& other) noexcept {
		if (&other == this)
			return *this;

		clear();
		if (keys)
			free(keys);

		amountUsed = other.amountUsed;
		capacity = other.capacity;
		keys = other.keys;
		mask = other.mask;
		other.amountUsed = 0;
		other.capacity = 0;
		other.keys = nullptr;
		other.mask = nullptr;

		return *this;
	}

	bool add(K key) override {
		int index = locate(key);

		if (index == -1) {
			resize(capacity * 2);
			index = locate(key);
		} else if (isPresent(index))
			return true;

		new (&keys[index]) K(key);
		setPresent(index, true);
		amountUsed++;

		return false;
	}

	/**
	 * @brief Adds all elements from another HashSet.
	 * @param other The source HashSet.
	 */
	void addFrom(const HashSet<K>& other) {
		for (uint32_t i = 0; i < other.capacity; i++) {
			if (other.isPresent(i))
				add(other.keys[i]);
		}
	}

	bool contains(K key) const override {
		int idx = locate(key);
		if (idx == -1)
			return false;
		return isPresent(idx);
	}

	bool remove(K key) override {
		int idx = locate(key);
		if (idx == -1 || !isPresent(idx))
			return false;

		keys[idx].~K();
		setPresent(idx, false);
		amountUsed--;

		// Fix any broken parts of the hashset
		idx = (idx + 1) % capacity;
		while (isPresent(idx)) {
			K tempKey = std::move(keys[idx]);
			keys[idx].~K();
			setPresent(idx, false);
			amountUsed--;

			add(std::move(tempKey));

			idx = (idx + 1) % capacity;
		}

		return true;
	}

	/**
	 * @brief Returns the current capacity of the hash set.
	 * @return The capacity.
	 */
	int getCapacity() const { return capacity; }

	/**
	 * @brief Checks if an element is present at the specified index.
	 * @param i The index to check.
	 * @return true if present, false otherwise.
	 */
	bool presentAtIndex(int i) const { return isPresent(i); }

	/**
	 * @brief Returns a reference to the key at the specified index.
	 * @param i The index.
	 * @return Reference to the key.
	 */
	K& keyAtIndex(int i) const { return keys[i]; }

	~HashSet() override {
		clear();
		if (keys)
			free(keys);
	}

	void clear() override {
		if (amountUsed == 0)
			return;
		for (unsigned int i = 0; i < capacity; i++) {
			if (isPresent(i)) {
				keys[i].~K();
			}
		}
		unsigned int maskSize = (capacity + 63) / 64;
		for (unsigned int i = 0; i < maskSize; i++)
			mask[i] = 0;
		amountUsed = 0;
	}

	/**
	 * @brief Checks if the hash set is full.
	 * @return true if amountUsed >= capacity, false otherwise.
	 */
	bool isFull() {
		return amountUsed >= capacity;
	}

	/**
	 * @brief Returns all elements as an ArrayList.
	 * @return ArrayList containing all elements of the set.
	 */
	ArrayList<K> toArrayList() const {
		ArrayList<K> out(amountUsed);

		for (uint32_t i = 0; i < capacity; i++)
			if (isPresent(i))
				out.add(keys[i]);

		return out;
	}

private:
	/**
	 * @brief Locates the index for a given key.
	 * @param key The key to locate.
	 * @return The index of the key, or -1 if not found.
	 */
	int locate(K key) const {
		// This number is probably prime.  I designed it to distribute the bits of the key around for
		// better key distribution.
		uint32_t hashedKey = (uint32_t)(key * 7224373213449699941LU);
		int startIndex = (hashedKey % capacity);
		int index = startIndex;
		while (isPresent(index)) {
			if (keys[index] == key)
				break;
			index = (index + 1) % capacity;
			if (index == startIndex)
				return -1;
		}
		return index;
	}

	/**
	 * @brief Resizes the hash set to a new capacity.
	 * @param newCapacity The new capacity.
	 */
	void resize(unsigned int newCapacity) {
		K* oldKeys = keys;
		uint64_t* oldMask = mask;
		unsigned int oldCapacity = capacity;

		allocate(newCapacity);
		capacity = newCapacity;

		// Copy everything over
		amountUsed = 0;
		for (unsigned int i = 0; i < oldCapacity; i++) {
			if ((oldMask[i >> 6] >> (i & 0x3F)) & 1) {
				add(std::move(oldKeys[i]));
				oldKeys[i].~K();
			}
		}

		// Free old memory
		free(oldKeys);
	}

	unsigned int capacity;
	unsigned int amountUsed {0};
	K* keys;
	uint64_t* mask;
};


#endif //EXCESSIVE_NUMBEREDMAP_H
