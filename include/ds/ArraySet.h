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


template<class T>
class ArraySet: public Set<T> {
public:

	ArraySet() {
		allocated = 64;
		length = 0;
		elements = (T*)malloc(sizeof(T) * allocated);
	}

	explicit ArraySet(int start_capacity) {
		allocated = start_capacity;
		length = 0;
		elements = (T*)malloc(sizeof(T) * allocated);
	}

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

	void addFrom(const ArrayList<T>& list) {
		int count = list.size();
		if (length + count > allocated) {
			allocated = allocated * 2 + (((count >> 4) + 1) << 4);
			elements = (T*)realloc(elements, sizeof(T) * allocated);
		}
		for (int i = 0; i < count; i++)
			addRaw(list.get(i));
	}

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

	void removeAt(int i) {
		memmove(&elements[i], &elements[i + 1], sizeof(T) * (--length - i));
	}

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

	inline int size() const { return length; }
	inline T& get(int i) const {
		if (i >= length || i < 0)
			printf("Out of bounds read of %i for ArraySet of length %i and allocated %i\n", i, length, allocated);
		return elements[i];
	}

	T* getMemory() const {
		return elements;
	}

	T* begin() override { return elements; }
	T* end() override { return &(elements[length]); }
	const T* begin() const override { return elements; }
	const T* end() const override { return &(elements[length]); }


	~ArraySet() override { free(elements); }

	inline void clear() override { length = 0; }

	T maximum() const {
		return elements[length - 1];
	}

	T minimum() const {
		return elements[0];
	}

protected:

private:

	// Adds elements without checking if there's enough memory
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
