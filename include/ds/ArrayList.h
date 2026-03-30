/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2023-04-02
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


#ifndef EXCESSIVE_ARRAYLIST_H
#define EXCESSIVE_ARRAYLIST_H


#include <new>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <utility>
#include "Range.h"
#include "MonkeyIterator.h"
#include "Container.h"

/*
 * This class was originally designed for use in larger embedded systems.
 * It uses malloc(), but only keeps one chunk of memory, and never shrinks it.
 * This prevents heap fragmentation.
 */

template<class T>
class ArrayList : public Container<T, T*, const T*> {
public:

	ArrayList() {
		allocated = 64;
		length = 0;
		elements = (T*)malloc(sizeof(T) * allocated);
	}

	explicit ArrayList(T&& v) {
		allocated = 64;
		length = 1;
		elements = (T*)malloc(sizeof(T) * allocated);
		new (elements) T(std::move(v));
	}

	explicit ArrayList(const T& v) {
		allocated = 64;
		length = 1;
		elements = (T*)malloc(sizeof(T) * allocated);
		new (elements) T(v);
	}

	explicit ArrayList(int start_capacity) {
		allocated = start_capacity;
		length = 0;
		elements = (T*)malloc(sizeof(T) * allocated);
	}

	ArrayList(const T* src, int size) {
		allocated = size;
		length = size;
		elements = (T*)malloc(sizeof(T) * allocated);
		for (int i = 0; i < size; i++)
			new (&elements[i]) T(src[i]);
	}

	explicit ArrayList(std::string_view view) {
		allocated = (int)view.length();
		length = (int)view.length();
		elements = (T*)malloc(sizeof(T) * allocated);
		memcpy(elements, view.data(), sizeof(T) * length);
	}

	ArrayList(ArrayList<T>&& from) noexcept
		: length(from.length), allocated(from.allocated), elements(from.exportMemory()) {}

	ArrayList(const ArrayList<T>& v) {
		allocated = v.allocated;
		length = v.length;
		elements = (T*)malloc(sizeof(T) * allocated);
		for (int i = 0; i < length; i++)
			new (&elements[i]) T(v.elements[i]);
	}

	ArrayList(std::initializer_list<T> initList) {
		allocated = static_cast<int>(initList.size());
		length = allocated;
		elements = (T*)malloc(sizeof(T) * allocated);

		int i = 0;
		for (const T& item : initList) {
			new (&elements[i++]) T(item);
		}
	}

	ArrayList& operator=(ArrayList<T>&& rhs) noexcept {
		clear();
		free((void*)elements);
		length = rhs.length;
		allocated = rhs.allocated;
		elements = rhs.exportMemory();
		return *this;
	}

	ArrayList& operator=(const ArrayList<T>& rhs) noexcept {
		if (&rhs != this) {
			clear();
			free((void*)elements);
			length = rhs.length;
			allocated = rhs.allocated;
			elements = (T*)malloc(allocated * sizeof(T));
			for (int i = 0; i < length; i++)
				new (&elements[i]) T(rhs.elements[i]);
		}
		return *this;
	}

	T& add(const T& item) {
		if (length == allocated) {
			allocated *= 2;
			if (allocated == 0) allocated = 8;
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}
		new (&elements[length]) T(item);
		return elements[length++];
	}

	T& add(T&& item) {
		if (length == allocated) {
			allocated *= 2;
			if (allocated == 0) allocated = 8;
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}
		new (&elements[length]) T(std::move(item));
		return elements[length++];
	}

	// Avoid using this function, it is slow
	T& addFirst(const T& item) {
		if (length == allocated) {
			allocated *= 2;
			if (allocated == 0) allocated = 8;
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}

		for (int i = length; i > 0; i--)
			elements[i] = std::move(elements[i - 1]);

		length++;
		new (&elements[0]) T(item);
		return elements[0];
	}

	void addMany(const T* values, int count) {
		if (length + count > allocated) {
			allocated = allocated * 2 + (((count >> 4) + 1) << 4);
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}
		for (int i = 0; i < count; i++)
			new (&elements[length + i]) T(values[i]);
		length += count;
	}

	void addMany(const ArrayList<T>& list) {
		int count = list.size();
		if (length + count > allocated) {
			allocated = allocated * 2 + (((count >> 4) + 1) << 4);
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}
		for (int i = 0; i < count; i++)
			new (&elements[length + i]) T(list.get(i));
		length += count;
	}

	void addMany(std::string_view view) {
		int count = (int)view.size();
		if (length + count > allocated) {
			allocated = allocated * 2 + (((count >> 4) + 1) << 4);
			elements = (T*)realloc(elements, sizeof(T) * allocated);
		}
		memcpy(&elements[length], view.data(), sizeof(T) * count);
		length += count;
	}

	void addCopies(T value, int count) {
		if (length + count > allocated) {
			allocated = allocated * 2 + (((count >> 4) + 1) << 4);
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}
		for (int i = 0; i < count; i++)
			new (&elements[length + i]) T(value);
		length += count;
	}

	void set(int i, const T& item) {
		if (i < length)
			elements[i] = item;
	}

	void set(int i, T&& item) {
		if (i < length)
			elements[i] = std::move(item);
	}

	void initialize(int i, const T& item) {
		if (i < length)
			new (&elements[i]) T(item);
	}

	void initialize(int i, T&& item) {
		if (i < length)
			new (&elements[i]) T(std::move(item));
	}

	T pop() {
		return std::move(elements[--length]);
	}

	void unorderedRemove(int i) {
		if (i < length && i >= 0)
			set(i, std::move(elements[--length]));
	}

	/**
	 * Will set the length to the new value.  If the new length is smaller, values after the
	 * new length are cut off.  If the new length is bigger, the new elements are initialized
	 * with zeros.
	 *
	 * If the capacity does not support the new size, then the capacity is set to double the
	 * new length.
	 * @param newLength New length for the array
	 */
	void resize(int newLength) {
		if (newLength > allocated) {
			allocated = newLength * 2;
			elements = (T*)realloc((void*)elements, sizeof(T) * allocated);
		}
		if (newLength > length) {
			void* toZero = (void*)&elements[length];
			size_t zeroSize = sizeof(T) * (newLength - length);
			bzero(toZero, zeroSize);
		}
		length = newLength;
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
			T* newElements = (T*)realloc((void*)elements, sizeof(T) * (size_t)size);
			if (newElements) {
				allocated = size;
				elements = newElements;
			} else
				return false;
		}

		return true;
	}

	ArrayList<T> subscriptInPlace(int a, int b = 0) {
		if (a < 0) a += length;
		if (b <= 0) b += length;

		return ArrayList<T>(&elements[a], b - a);
	}

	inline int size() const { return length; }
	inline T& get(int i) const {
		if (i >= length || i < 0) {
			printf("Out of bounds read of %i for ArrayList of length %i and allocated %i\n", i, length, allocated);
		}
		return elements[i];
	}

	T* getMemory() const {
		return elements;
	}

	/**
	 * Sets the memory of this object to nullptr, and returns a pointer to the original memory.
	 * This must be the LAST function called on the object, as most functions will segfault after
	 * this is called.  This is used to make the deconstructor not delete the allocated memory, so it
	 * can be used elsewhere, and transfers ownership of the memory to the caller.
	 *
	 * This also sets the length of the array to zero, so if you need to know the array size, call
	 * size() before calling this.
	 *
	 * @return The memory contents
	 */
	T* exportMemory() {
		T* tmp = elements;
		elements = nullptr;
		length = 0;
		return tmp;
	}

	T* begin() override { return elements; }
	T* end() override { return &(elements[length]); }
	const T* begin() const override { return elements; }
	const T* end() const override { return &(elements[length]); }

	template<class E>
	Range<MonkeyIterator<T*, E>> rangeOf() {
		return Range<MonkeyIterator<T*, E>>(
				MonkeyIterator<T*, E>(elements),
				MonkeyIterator<T*, E>(&elements[length])
		);
	}

	template<class E>
	Range<MonkeyIterator<T*, E>> rangeOf(int start, int end) {
		return Range<MonkeyIterator<T*, E>>(
				MonkeyIterator<T*, E>(&elements[start]),
				MonkeyIterator<T*, E>(&elements[end])
		);
	}

	inline void clear() {
		for (int i = 0; i < length; i++)
			elements[i].~T();
		length = 0;
	}

	void addString(const T* s) {
		addMany(s, strlen((const char*)s));
	}

	~ArrayList() {
		clear();
		free((void*)elements);
	}

protected:

private:
	int length, allocated;
	T* elements;
};



#endif //EXCESSIVE_ARRAYLIST_H
