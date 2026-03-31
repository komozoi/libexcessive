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


#ifndef EXCESSIVE_HASHMAP_H
#define EXCESSIVE_HASHMAP_H


#include "stdint.h"
#include "alloc/Allocator.h"
#include "Map.h"
#include <new>


template <class K, class T>
class HashMap: public Map<K, T> {
protected:
	void allocate(unsigned int cap) {
		unsigned int maskSize = (cap + 63) / 64;
		size_t totalSize = cap * sizeof(K) + cap * sizeof(T) + maskSize * sizeof(uint64_t);
		keys = (K*)allocator.alloc(totalSize);
		values = (T*)(keys + cap);
		mask = (uint64_t*)(values + cap);
		for (unsigned int i = 0; i < maskSize; i++)
			mask[i] = 0;
	}

	bool isPresent(unsigned int index) const {
		return (mask[index >> 6] >> (index & 0x3F)) & 1;
	}

	void setPresent(unsigned int index, bool present) {
		if (present)
			mask[index >> 6] |= (1ULL << (index & 0x3F));
		else
			mask[index >> 6] &= ~(1ULL << (index & 0x3F));
	}

public:
	explicit HashMap(unsigned int capacity, Allocator& allocator = defaultAllocator) : capacity(capacity), allocator(allocator) {
		if (this->capacity == 0)
			this->capacity = 1;
		allocate(this->capacity);
	}

	template<typename U1, typename U2, typename U3>
	HashMap(const Container<MapElement<K, T>, U1, U2, U3>& other) : capacity(other.getCapacity()), amountUsed(0), allocator(other.allocator) {
		allocate(capacity);
		putFrom(other);
	}

	HashMap(HashMap<K, T>&& other) noexcept
		: capacity(other.capacity), amountUsed(other.amountUsed), keys(other.keys), values(other.values), mask(other.mask), allocator(other.allocator) {
		other.keys = nullptr;
		other.values = nullptr;
		other.mask = nullptr;
		other.capacity = 0;
		other.amountUsed = 0;
	}

	HashMap(const HashMap<K, T>& other) : capacity(other.capacity), amountUsed(0), allocator(other.allocator) {
		allocate(capacity);
		for (unsigned int i = 0; i < capacity; i++) {
			if (other.isPresent(i)) {
				new (&keys[i]) K(other.keys[i]);
				new (&values[i]) T(other.values[i]);
				setPresent(i, true);
				amountUsed++;
			}
		}
	}

	virtual HashMap<K, T>& operator=(const HashMap<K, T>& other) {
		if (&other != this) {
			clear();
			if (keys) {
				allocator.free(keys);
			}

			allocator = other.allocator;
			capacity = other.capacity;
			allocate(capacity);

			for (unsigned int i = 0; i < capacity; i++) {
				if (other.isPresent(i)) {
					new (&keys[i]) K(other.keys[i]);
					new (&values[i]) T(other.values[i]);
					setPresent(i, true);
				}
			}

			amountUsed = other.amountUsed;
		}
		return *this;
	}

	virtual HashMap<K, T>& operator=(HashMap<K, T>&& other) noexcept {
		if (&other != this) {
			clear();
			if (keys) {
				allocator.free(keys);
			}

			keys = other.keys;
			values = other.values;
			mask = other.mask;
			capacity = other.capacity;
			amountUsed = other.amountUsed;
			allocator = other.allocator;

			other.keys = nullptr;
			other.values = nullptr;
			other.mask = nullptr;
			other.capacity = 0;
			other.amountUsed = 0;
		}
		return *this;
	}

	MapElement<K, T> getFirst(long& index) override {
		for (unsigned int i = 0; i < capacity; i++) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements in the map");
	}

	MapElement<K, T> getLast(long& index) override {
		for (int i = (int)capacity - 1; i >= 0; i--) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements in the map");
	}

	MapElement<K, T> getNext(long& index) override {
		if (index < 0) return getFirst(index);
		for (unsigned int i = index + 1; i < capacity; i++) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements left");
	}

	MapElement<K, T> getPrevious(long& index) override {
		if (index < 0) return getLast(index);
		for (int i = (int)index - 1; i >= 0; i--) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements left");
	}

	MapElement<K, T> getElementAtIndex(long index) override {
		if (index >= 0 && index < (long)capacity && isPresent(index)) {
			return {keys[index], values[index]};
		}
		throw std::out_of_range("Invalid index");
	}

	const MapElement<K, T> getFirst(long& index) const override {
		for (unsigned int i = 0; i < capacity; i++) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements in the map");
	}

	const MapElement<K, T> getLast(long& index) const override {
		for (int i = (int)capacity - 1; i >= 0; i--) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements in the map");
	}

	const MapElement<K, T> getNext(long& index) const override {
		if (index < 0) return getFirst(index);
		for (unsigned int i = index + 1; i < capacity; i++) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements left");
	}

	const MapElement<K, T> getPrevious(long& index) const override {
		if (index < 0) return getLast(index);
		for (int i = (int)index - 1; i >= 0; i--) {
			if (isPresent(i)) {
				index = i;
				return {keys[i], values[i]};
			}
		}
		throw std::out_of_range("No elements left");
	}

	const MapElement<K, T> getElementAtIndex(long index) const override {
		if (index >= 0 && index < (long)capacity && isPresent(index)) {
			return {keys[index], values[index]};
		}
		throw std::out_of_range("Invalid index");
	}

	bool hasKey(K key) const override {
		int idx = locate(key);
		if (idx == -1)
			return false;
		return isPresent(idx);
	}

	T* getPtr(K key) override {
		int idx = locate(key);
		if (idx == -1 || !isPresent(idx))
			return nullptr;
		return &values[idx];
	}

	const T* getPtr(K key) const override {
		int idx = locate(key);
		if (idx == -1 || !isPresent(idx))
			return nullptr;
		return &values[idx];
	}

	T get(K key) const override {
		int idx = locate(key);
		if (idx == -1 || !isPresent(idx))
			return T();
		return values[idx];
	}

	T getOrDefault(K key, T defaultValue) const override {
		int idx = locate(key);
		if (idx == -1 || !isPresent(idx))
			return defaultValue;
		return values[idx];
	}

	T remove(K key) override {
		int idx = locate(key);
		if (idx == -1 || !isPresent(idx))
			return T();

		T oldValue = std::move(values[idx]);
		values[idx].~T();
		keys[idx].~K();
		setPresent(idx, false);
		amountUsed--;

		// Fix any broken parts of the hashmap
		idx = (idx + 1) % capacity;
		while (isPresent(idx)) {
			K tempKey = std::move(keys[idx]);
			T tempValue = std::move(values[idx]);
			keys[idx].~K();
			values[idx].~T();
			setPresent(idx, false);
			amountUsed--;

			put(std::move(tempKey), std::move(tempValue));

			idx = (idx + 1) % capacity;
		}

		return oldValue;
	}

	virtual bool remove(K key, T& out) override {
		int idx = locate(key);
		if (idx == -1 || !isPresent(idx))
			return false;

		out = std::move(values[idx]);
		values[idx].~T();
		keys[idx].~K();
		setPresent(idx, false);
		amountUsed--;

		// Fix any broken parts of the hashmap
		idx = (idx + 1) % capacity;
		while (isPresent(idx)) {
			K tempKey = std::move(keys[idx]);
			T tempValue = std::move(values[idx]);
			keys[idx].~K();
			values[idx].~T();
			setPresent(idx, false);
			amountUsed--;

			put(std::move(tempKey), std::move(tempValue));

			idx = (idx + 1) % capacity;
		}

		return true;
	}

	virtual T& putPtr(K key, T* value) override {
		int index = locate(key);
		if ((index < 0) || (index >= (int)capacity)) {
			abort();
		}

		if (!isPresent(index)) {
			new (&keys[index]) K(key);
			new (&values[index]) T(std::move(*value));
			delete value;
			setPresent(index, true);
			amountUsed++;
		} else {
			values[index] = std::move(*value);
			delete value;
		}

		if (amountUsed == capacity) {
			resize(capacity * 2);
			index = locate(key);
			if ((index < 0) || (index >= (int)capacity))
				abort();
		}

		return values[index];
	}

	virtual T& put(K key, const T& value) override {
		int index = locate(key);
		if ((index < 0) || (index >= (int)capacity))
			abort();

		if (!isPresent(index)) {
			new (&keys[index]) K(key);
			new (&values[index]) T(value);
			setPresent(index, true);
			amountUsed++;
		} else
			values[index] = value;

		if (amountUsed == capacity) {
			resize(capacity * 2);
			index = locate(key);
			if ((index < 0) || (index >= (int)capacity))
				abort();
		}

		return values[index];
	}

	T& put(K key, T&& value) override {
		int index = locate(key);
		if ((index < 0) || (index >= (int)capacity)) {
			abort();
		}

		if (!isPresent(index)) {
			new (&keys[index]) K(key);
			new (&values[index]) T(std::move(value));
			setPresent(index, true);
			amountUsed++;
		} else
			values[index] = std::move(value);

		if (amountUsed == capacity) {
			resize(capacity * 2);
			index = locate(key);
			if ((index < 0) || (index >= (int)capacity))
				abort();
		}

		return values[index];
	}

	template<typename U1, typename U2, typename U3>
	void putFrom(const Container<MapElement<K, T>, U1, U2, U3>& other) {
		for (MapElement<K, T> e : other)
			put(e.key, e.value);
	}

	int getCapacity() const { return capacity; }
	int size() const { return amountUsed; }
	bool presentAtIndex(int i) const { return isPresent(i); }
	const T& valueAtIndex(int i) const { return values[i]; }
	const K& keyAtIndex(int i) const { return keys[i]; }
	T& valueAtIndex(int i) { return values[i]; }
	K& keyAtIndex(int i) { return keys[i]; }

	~HashMap() override {
		clear();
		if (keys)
			allocator.free(keys);
	}

	void clear() override {
		if (amountUsed == 0)
			return;
		for (unsigned int i = 0; i < capacity; i++) {
			if (isPresent(i)) {
				keys[i].~K();
				values[i].~T();
			}
		}
		unsigned int maskSize = (capacity + 63) / 64;
		for (unsigned int i = 0; i < maskSize; i++)
			mask[i] = 0;
		amountUsed = 0;
	}

	bool isFull() {
		return amountUsed >= capacity;
	}

protected:
	template<class U>
	int locate(const U& key) const {
		uint32_t hashedKey = (uint32_t)(((uint64_t)key * 7224373213449699941LU) >> 32);
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

	template<class A, class B>
	int locate(const std::pair<A,B>& key) const {
		uint32_t h1 = (uint32_t)(((uint64_t)key.first  * 7224373213449699941LU) >> 32);
		uint32_t h2 = (uint32_t)(((uint64_t)key.second * 23428012901LU) >> 2);

		uint32_t hashedKey = h1 ^ (h2 * 91);

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

	virtual void resize(unsigned int newCapacity) {
		K* oldKeys = keys;
		T* oldValues = values;
		uint64_t* oldMask = mask;
		unsigned int oldCapacity = capacity;
		unsigned int oldUsed = amountUsed;

		allocate(newCapacity);
		capacity = newCapacity;
		amountUsed = 0;

		// Copy everything over
		for (unsigned int i = 0; i < oldCapacity; i++) {
			if ((oldMask[i >> 6] >> (i & 0x3F)) & 1) {
				put(std::move(oldKeys[i]), std::move(oldValues[i]));
				oldKeys[i].~K();
				oldValues[i].~T();
			}
		}
		if (oldUsed != amountUsed)
			abort();

		// Free old memory
		allocator.free(oldKeys);
	}

	unsigned int capacity;
	unsigned int amountUsed {0};
	K* keys;
	T* values;
	uint64_t* mask;

	Allocator& allocator;
};


#endif //EXCESSIVE_HASHMAP_H
