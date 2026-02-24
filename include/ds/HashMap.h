//
// Created by nuclaer on 2/11/23.
//

#ifndef EXCESSIVE_HASHMAP_H
#define EXCESSIVE_HASHMAP_H


#include "stdio.h"
#include "stdint.h"
#include "alloc/Allocator.h"
#include "Map.h"


template <class K, class T>
class HashMap: public Map<K, T> {
protected:
	struct hashmap_entry_s {
		K key;
		T* value;
		bool present;
	};

public:
	explicit HashMap(unsigned int capacity, Allocator& allocator = defaultAllocator) : capacity(capacity), allocator(allocator) {
		if (this->capacity == 0)
			this->capacity = 1;
		entries = allocator.allocateArray<hashmap_entry_s>(this->capacity);
		for (unsigned int i = 0; i < this->capacity; i++)
			entries[i].present = false;
	}

	HashMap(const HashMap<K, T>& other) : capacity(other.capacity), amountUsed(other.amountUsed), allocator(other.allocator) {
		entries = allocator.allocateArray<hashmap_entry_s>(capacity);
		for (unsigned int i = 0; i < capacity; i++)
			if ((entries[i].present = other.entries[i].present)) {
				new (&entries[i].key) K(other.entries[i].key);
				entries[i].value = new T(*other.entries[i].value);
			}
	}

	HashMap(HashMap<K, T>&& other) noexcept
		: capacity(other.capacity), amountUsed(other.amountUsed), entries(other.entries), allocator(other.allocator) {
		other.entries = nullptr;
		other.capacity = 0;
		other.amountUsed = 0;
	}

	virtual HashMap<K, T>& operator=(const HashMap<K, T>& other) {
		if (&other != this) {
			if (entries) {
				clear();
				allocator.free(entries);
			}

			allocator = other.allocator;
			entries = allocator.allocateArray<hashmap_entry_s>(capacity = other.capacity);

			for (unsigned int i = 0; i < capacity; i++)
				if ((entries[i].present = other.entries[i].present)) {
					new (&entries[i].key) K(other.entries[i].key);
					entries[i].value = new T(*other.entries[i].value);
				}

			amountUsed = other.amountUsed;
		}
		return *this;
	}

	HashMap<K, T>& operator=(HashMap<K, T>&& other) noexcept {
		if (entries) {
			clear();
			allocator.free(entries);
		}
		entries = other.entries;
		capacity = other.capacity;
		amountUsed = other.amountUsed;
		allocator = other.allocator;
		other.entries = nullptr;
		other.capacity = 0;
		other.amountUsed = 0;
		return *this;
	}

	bool hasKey(K key) const override {
		int idx = locate(key);
		if (idx == -1)
			return false;
		return entries[idx].present;
	}

	T* getPtr(K key) override {
		int idx = locate(key);
		if (idx == -1 || !entries[idx].present)
			return nullptr;
		return entries[idx].value;
	}

	const T* getPtr(K key) const override {
		int idx = locate(key);
		if (idx == -1 || !entries[idx].present)
			return nullptr;
		return entries[idx].value;
	}

	T get(K key) const override {
		int idx = locate(key);
		if (idx == -1 || !entries[idx].present)
			return T();
		return *entries[idx].value;
	}

	T getOrDefault(K key, T defaultValue) const override {
		int idx = locate(key);
		if (idx == -1 || !entries[idx].present)
			return defaultValue;
		return *entries[idx].value;
	}

	T remove(K key) override {
		int idx = locate(key);
		if (idx == -1 || !entries[idx].present)
			return T();

		T oldValue = std::move(*entries[idx].value);
		delete entries[idx].value;
		entries[idx].key.~K();
		entries[idx].present = false;
		amountUsed--;

		// Fix any broken parts of the hashmap
		idx = (idx + 1) % capacity;
		while (entries[idx].present) {
			entries[idx].present = false;
			amountUsed--;

			putPtr(std::move(entries[idx].key), entries[idx].value);

			idx = (idx + 1) % capacity;
		}

		return oldValue;
	}

	virtual bool remove(K key, T& out) override {
		int idx = locate(key);
		if (idx == -1 || !entries[idx].present)
			return false;

		out = std::move(*entries[idx].value);
		delete entries[idx].value;
		entries[idx].key.~K();
		entries[idx].present = false;
		amountUsed--;

		// Fix any broken parts of the hashmap
		idx = (idx + 1) % capacity;
		while (entries[idx].present) {
			entries[idx].present = false;
			amountUsed--;

			putPtr(std::move(entries[idx].key), entries[idx].value);

			idx = (idx + 1) % capacity;
		}

		return true;
	}

	virtual T& putPtr(K key, T* value) override {
		int index = locate(key);
		if ((index < 0) || (index >= (int)capacity)) {
			abort();
		}

		if (!entries[index].present) {
			new (&entries[index].key) K(key);
			entries[index].value = value;
			entries[index].present = true;
			amountUsed++;
		} else {
			delete entries[index].value;
			entries[index].value = value;
		}

		if (amountUsed == capacity) {
			resize(capacity * 2);
			index = locate(key);
			if ((index < 0) || (index >= (int)capacity))
				abort();
		}

		return *entries[index].value;
	}

	virtual T& put(K key, const T& value) override {
		int index = locate(key);
		if ((index < 0) || (index >= (int)capacity))
			abort();

		if (!entries[index].present) {
			new (&entries[index].key) K(key);
			entries[index].value = new T(value);
			entries[index].present = true;
			amountUsed++;
		} else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
			*entries[index].value = value;
#pragma GCC diagnostic pop

		if (amountUsed == capacity) {
			resize(capacity * 2);
			index = locate(key);
			if ((index < 0) || (index >= (int)capacity))
				abort();
		}

		return *entries[index].value;
	}

	virtual T& put(K key, T&& value) override {
		int index = locate(key);
		if ((index < 0) || (index >= (int)capacity)) {
			abort();
		}

		if (!entries[index].present) {
			new (&entries[index].key) K(key);
			entries[index].value = new T(std::move(value));
			entries[index].present = true;
			amountUsed++;
		} else
			*entries[index].value = std::move(value);

		if (amountUsed == capacity) {
			resize(capacity * 2);
			index = locate(key);
			if ((index < 0) || (index >= (int)capacity))
				abort();
		}

		return *entries[index].value;
	}

	virtual void putFrom(const HashMap<K, T>& other) {
		for (uint32_t i = 0; i < other.capacity; i++) {
			hashmap_entry_s& entry = other.entries[i];
			if (entry.present)
				put(entry.key, *entry.value);
		}
	}

	int getCapacity() const { return capacity; }
	unsigned int numUsed() const { return amountUsed; }
	bool presentAtIndex(int i) const { return entries[i].present; }
	const T& valueAtIndex(int i) const { return *entries[i].value; }
	const K& keyAtIndex(int i) const { return entries[i].key; }
	T& valueAtIndex(int i) { return *entries[i].value; }
	K& keyAtIndex(int i) { return entries[i].key; }

	~HashMap() override {
		clear();
		allocator.free(entries);
	}

	void clear() override {
		if (amountUsed == 0)
			return;
		for (unsigned int i = 0; i < capacity; i++) {
			if (entries[i].present) {
				entries[i].key.~K();
				delete entries[i].value;
				entries[i].present = false;
			}
		}
		amountUsed = 0;
	}

	bool isFull() {
		return amountUsed >= capacity;
	}

protected:
	int locate(const K& key) const {
		// This number is probably prime.  I designed it to distribute the bits of the key around for
		// better key distribution.
		uint32_t hashedKey = (uint32_t)(((uint64_t)key * 7224373213449699941LU) >> 32);
		int startIndex = (hashedKey % capacity);
		int index = startIndex;
		while (entries[index].present) {
			if (entries[index].key == key)
				break;
			index = (index + 1) % capacity;
			if (index == startIndex)
				return -1;
		}
		return index;
	}

	virtual void resize(unsigned int newCapacity) {
		hashmap_entry_s* oldEntries = entries;
		unsigned int oldCapacity = capacity;
		unsigned int oldUsed = amountUsed;

		entries = allocator.allocateArray<hashmap_entry_s>(newCapacity);
		capacity = newCapacity;
		amountUsed = 0;
		for (unsigned int i = 0; i < newCapacity; i++)
			entries[i].present = false;

		// Copy everything over
		for (unsigned int i = 0; i < oldCapacity; i++) {
			if (oldEntries[i].present)
				putPtr(oldEntries[i].key, oldEntries[i].value);
		}
		if (oldUsed != amountUsed)
			abort();

		// Free old memory
		allocator.free(oldEntries);
	}

	unsigned int capacity;
	unsigned int amountUsed {0};
	hashmap_entry_s* entries;

	Allocator& allocator;
};


#endif //EXCESSIVE_HASHMAP_H
