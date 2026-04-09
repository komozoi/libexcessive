/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2024-01-03
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


#ifndef EXCESSIVE_ARRAYSET_H
#define EXCESSIVE_ARRAYSET_H

#include <new>
#include <utility>
#include <type_traits>
#include "ArrayList.h"
#include "Set.h"


/**
 * @brief A set implementation using a sorted dynamic array.
 *
 * This class maintains elements in sorted order to allow binary search
 * for contains and removal operations. Insertions take O(n) time.
 *
 * @tparam T The type of elements in the set.
 */
template<class T>
class ArraySet: public Set<T> {
public:

	/**
	 * @brief Constructs an empty ArraySet with a default initial capacity.
	 */
	ArraySet() {
		allocated = 64;
		length = 0;
		elements = (T*)malloc(sizeof(T) * allocated);
	}

	/**
	 * @brief Constructs an empty ArraySet with the specified initial capacity.
	 * @param start_capacity Initial capacity.
	 */
	explicit ArraySet(int start_capacity) {
		allocated = start_capacity;
		length = 0;
		elements = (T*)malloc(sizeof(T) * allocated);
	}

	/**
	 * @brief Copy constructor.
	 * @param other The ArraySet to copy from.
	 */
	ArraySet(const ArraySet<T>& other) {
		allocated = other.allocated;
		length = other.length;
		elements = (T*)malloc(sizeof(T) * allocated);
		for (int i = 0; i < length; i++)
			new (&elements[i]) T(other.elements[i]);
	}

	/**
	 * @brief Move constructor.
	 * @param other The ArraySet to move from.
	 */
	ArraySet(ArraySet<T>&& other) noexcept {
		allocated = other.allocated;
		length = other.length;
		elements = other.elements;
		other.allocated = 0;
		other.length = 0;
		other.elements = nullptr;
	}

	/**
	 * @brief Copy assignment operator.
	 * @param other The ArraySet to copy from.
	 * @return Reference to this ArraySet.
	 */
	ArraySet<T>& operator=(const ArraySet<T>& other) {
		if (this != &other) {
			clear();
			free(elements);
			allocated = other.allocated;
			length = other.length;
			elements = (T*)malloc(sizeof(T) * allocated);
			for (int i = 0; i < length; i++)
				new (&elements[i]) T(other.elements[i]);
		}
		return *this;
	}

	/**
	 * @brief Move assignment operator.
	 * @param other The ArraySet to move from.
	 * @return Reference to this ArraySet.
	 */
	ArraySet<T>& operator=(ArraySet<T>&& other) noexcept {
		if (this != &other) {
			clear();
			free(elements);
			allocated = other.allocated;
			length = other.length;
			elements = other.elements;
			other.allocated = 0;
			other.length = 0;
			other.elements = nullptr;
		}
		return *this;
	}

	/**
	 * @brief Constructs an ArraySet from a raw array.
	 * @param src Pointer to the source array.
	 * @param size Number of elements to add.
	 */
	ArraySet(const T* src, int size) {
		allocated = size;
		length = 0;
		elements = (T*)malloc(sizeof(T) * allocated);
		Set<T>::addMany(src, size);
	}

	bool add(const T& item) override {
		if (length == allocated) {
			int newAllocated = allocated * 2;
			if (newAllocated == 0) newAllocated = 8;
			prepare(newAllocated);
		}
		return addRaw(item);
	}

	/**
	 * @brief Adds elements from another container.
	 * @param container the source container
	 */
	template<class U>
	void addFrom(const U& container) {
		int count = container.size();
		if (length + count > allocated) {
			int newAllocated = allocated * 2 + (((count >> 4) + 1) << 4);
			prepare(newAllocated);
		}
		for (const T& item : container)
			addRaw(item);
	}

	bool contains(const T& query) const override {
		return search(query, false) != -1;
	}

	/**
	 * @brief Removes and returns the last element.
	 * @return The removed element.
	 */
	T pop() {
		T result = std::move(elements[--length]);
		elements[length].~T();
		return result;
	}

	bool remove(const T& key) override {
		int i = search(key);
		if (i == -1)
			return false;
		removeAt(i);
		return true;
	}

	/**
	 * @brief Removes the element at the specified index.
	 * @param i Index of the element to remove.
	 */
	void removeAt(int i) {
		if constexpr (std::is_trivially_copyable<T>::value) {
			memmove(&elements[i], &elements[i + 1], sizeof(T) * (--length - i));
		} else {
			elements[i].~T();
			for (int j = i; j < length - 1; j++) {
				new(&elements[j]) T(std::move(elements[j + 1]));
				elements[j + 1].~T();
			}
			length--;
		}
	}

	/**
	 * @brief Searches for an element using binary search.
	 * @param query The element to search for.
	 * @param returnNearest If true, returns the insertion index if not found.
	 * @return The index of the element, or -1 (or insertion index) if not found.
	 */
	int search(const T& query, bool returnNearest = false) const {
		int low = 0;
		int high = length - 1;
		int result = 0;

		while (low <= high) {
			int mid = low + (high - low) / 2;
			if (elements[mid] == query) {
				return mid;
			} else if (elements[mid] < query) {
				low = mid + 1;
				result = mid + 1; // Keep track of the next-highest value
			} else {
				high = mid - 1;
			}
		}
		return returnNearest ? result : -1;
	}

	/**
	 * If the allocated elements of the array is less than `size`, then the array is reallocated to match the new
	 * size.  This function will never change the length of the array, only how many elements are allocated.  It will
	 * never reduce the allocated memory or deallocate it.
	 * @param size The minimum number of elements the array should occupy in memory
	 * @return true on success, false on failure (if realloc returned nullptr)
	 */
	bool prepare(int size) {
		if (size > allocated) {
			T* newElements = (T*)malloc(sizeof(T) * (size_t)size);
			if (newElements) {
				for (int i = 0; i < length; i++) {
					new (&newElements[i]) T(std::move(elements[i]));
					elements[i].~T();
				}
				free(elements);
				allocated = size;
				elements = newElements;
			} else
				return false;
		}

		return true;
	}

	inline int size() const override { return length; }

	/**
	 * @brief Gets the element at the specified index.
	 * @param i Index of the element.
	 * @return Reference to the element.
	 */
	inline T& get(int i) const {
		if (i >= length || i < 0)
			printf("Out of bounds read of %i for ArraySet of length %i and allocated %i\n", i, length, allocated);
		return elements[i];
	}

	/**
	 * @brief Returns a pointer to the internal memory buffer.
	 * @return Pointer to the elements.
	 */
	T* getMemory() const {
		return elements;
	}

	T* begin() override { return elements; }
	T* end() override { return &(elements[length]); }
	const T* begin() const override { return elements; }
	const T* end() const override { return &(elements[length]); }


	~ArraySet() override {
		clear();
		free(elements);
	}

	inline void clear() override {
		for (int i = 0; i < length; i++)
			elements[i].~T();
		length = 0;
	}

	/**
	 * @brief Returns the maximum element in the set.
	 * @return The maximum element.
	 */
	T maximum() const {
		return elements[length - 1];
	}

	/**
	 * @brief Returns the minimum element in the set.
	 * @return The minimum element.
	 */
	T minimum() const {
		return elements[0];
	}

private:

	/**
	 * @brief Adds an element without checking if there's enough memory.
	 * @param item The element to add.
	 * @return true if already present, false otherwise.
	 */
	bool addRaw(const T& item) {
		int insertionIndex = search(item, true);
		if (insertionIndex < length) {
			if (elements[insertionIndex] == item)
				return true;
			if constexpr (std::is_trivially_copyable<T>::value) {
				memmove(&elements[insertionIndex + 1], &elements[insertionIndex], sizeof(T) * (length - insertionIndex));
			} else {
				for (int i = length; i > insertionIndex; i--) {
					new(&elements[i]) T(std::move(elements[i - 1]));
					elements[i - 1].~T();
				}
			}
		}
		new(&elements[insertionIndex]) T(item);
		length++;

		return false;
	}

	int length, allocated;
	T* elements;
};

#endif //EXCESSIVE_ARRAYSET_H
