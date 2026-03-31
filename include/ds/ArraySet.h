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

	bool add(T item) override {
		if (length == allocated) {
			allocated *= 2;
			if (allocated == 0) allocated = 8;
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}
		return addRaw(item);
	}

	/**
	 * @brief Adds all elements from an ArrayList.
	 * @param list The source ArrayList.
	 */
	void addFrom(const ArrayList<T>& list) {
		int count = list.size();
		if (length + count > allocated) {
			allocated = allocated * 2 + (((count >> 4) + 1) << 4);
			elements = (T*)realloc(elements, sizeof(T) * allocated);
		}
		for (int i = 0; i < count; i++)
			addRaw(list.get(i));
	}

	/**
	 * @brief Adds all elements from another ArraySet.
	 * @param list The source ArraySet.
	 */
	void addFrom(const ArraySet<T>& list) {
		int count = list.size();
		if (length + count > allocated) {
			allocated = allocated * 2 + (((count >> 4) + 1) << 4);
			elements = (T*)realloc(elements, sizeof(T) * allocated);
		}
		for (int i = 0; i < count; i++)
			addRaw(list.get(i));
	}

	bool contains(T query) const override {
		return search(query, false) != -1;
	}

	/**
	 * @brief Removes and returns the last element.
	 * @return The removed element.
	 */
	T pop() {
		return elements[--length];
	}

	bool remove(T key) override {
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
		memmove(&elements[i], &elements[i + 1], sizeof(T) * (--length - i));
	}

	/**
	 * @brief Searches for an element using binary search.
	 * @param query The element to search for.
	 * @param returnNearest If true, returns the insertion index if not found.
	 * @return The index of the element, or -1 (or insertion index) if not found.
	 */
	int search(T query, bool returnNearest = false) const {
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
			T* newElements = (T*)realloc(elements, sizeof(T) * (size_t)size);
			if (newElements) {
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


	~ArraySet() override { free(elements); }

	inline void clear() override { length = 0; }

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

protected:

private:

	/**
	 * @brief Adds an element without checking if there's enough memory.
	 * @param item The element to add.
	 * @return true if already present, false otherwise.
	 */
	bool addRaw(T item) {
		int insertionIndex = search(item, true);
		if (insertionIndex < length) {
			if (elements[insertionIndex] == item)
				return true;
			memmove(&elements[insertionIndex + 1], &elements[insertionIndex], sizeof(T) * (length - insertionIndex));
		}
		elements[insertionIndex] = item;
		length++;

		return false;
	}

	int length, allocated;
	T* elements;
};

#endif //EXCESSIVE_ARRAYSET_H
