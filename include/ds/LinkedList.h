/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2023-01-08
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


#ifndef EXCESSIVE_LINKEDLIST_H
#define EXCESSIVE_LINKEDLIST_H


#include <cstdlib>
#include <cstdio>
#include <iterator>
#include <cstddef>
#include "Container.h"


template<class T>
class LinkedList;


template<class T>
class linkedlist_value_container_t {
public:
	explicit inline linkedlist_value_container_t(const T& value) : value(value), next(nullptr), previous(nullptr) {}

	T value;
	linkedlist_value_container_t* next;
	linkedlist_value_container_t* previous;

	class Iterator {
	public:
		// Needed for reverse_iterator
		typedef std::bidirectional_iterator_tag iterator_category;
		typedef T value_type;
		typedef std::ptrdiff_t difference_type;
		typedef T* pointer;
		typedef T& reference;

		Iterator() : node(nullptr), atEnd(true) {}
		Iterator(linkedlist_value_container_t* node, bool atEnd) : node(node), atEnd(atEnd) {}

		T& operator*() const {
			return node->value;
		}

		T* operator->() const {
			if (atEnd)
				return nullptr;
			return &node->value;
		}

		Iterator& operator++() {
			if (!atEnd && node) {
				if (node->next)
					node = node->next;
				else
					atEnd = true;
			}
			return *this;
		}

		Iterator operator++(int) {
			Iterator tmp = *this;
			++(*this);
			return tmp;
		}

		Iterator& operator--() {
			if (atEnd)
				atEnd = false;
			else
				node = node ? node->previous : nullptr;
			return *this;
		}

		Iterator operator--(int) {
			Iterator tmp = *this;
			--(*this);
			return tmp;
		}

		bool operator==(const Iterator& other) const {
			if (atEnd != other.atEnd) return false;
			if (atEnd) return true; // Both are at end, node doesn't matter (or should both be last)
			return node == other.node;
		}
		bool operator!=(const Iterator& other) const { return !(*this == other); }

	private:
		linkedlist_value_container_t* node;
		bool atEnd;
	};

	class ConstIterator {
	public:
		// Needed for reverse_iterator
		typedef std::bidirectional_iterator_tag iterator_category;
		typedef const T value_type;
		typedef std::ptrdiff_t difference_type;
		typedef const T* pointer;
		typedef const T& reference;

		ConstIterator() : node(nullptr), atEnd(true) {}
		ConstIterator(const linkedlist_value_container_t* node, bool atEnd) : node(node), atEnd(atEnd) {}
		ConstIterator(const Iterator& other) : node(other.node), atEnd(other.atEnd) {}

		const T& operator*() const { return node->value; }
		const T* operator->() const {
			if (atEnd)
				return nullptr;
			return &node->value;
		}

		ConstIterator& operator++() {
			if (!atEnd && node) {
				if (node->next)
					node = node->next;
				else
					atEnd = true;
			}
			return *this;
		}

		ConstIterator operator++(int) {
			ConstIterator tmp = *this;
			++(*this);
			return tmp;
		}

		ConstIterator& operator--() {
			if (atEnd)
				atEnd = false;
			else
				node = node ? node->previous : nullptr;
			return *this;
		}

		ConstIterator operator--(int) {
			ConstIterator tmp = *this;
			--(*this);
			return tmp;
		}

		bool operator==(const ConstIterator& other) const {
			if (atEnd != other.atEnd) return false;
			if (atEnd) return true;
			return node == other.node;
		}
		bool operator!=(const ConstIterator& other) const { return !(*this == other); }

	private:
		const linkedlist_value_container_t* node;
		bool atEnd;
	};
};


template<class T>
class LinkedList : public Container<T, typename linkedlist_value_container_t<T>::Iterator, typename linkedlist_value_container_t<T>::ConstIterator> {
public:

	typedef typename linkedlist_value_container_t<T>::Iterator iterator;
	typedef typename linkedlist_value_container_t<T>::ConstIterator const_iterator;

	iterator begin() override { return iterator(first, first == nullptr); }
	iterator end() override { return iterator(last, true); }
	const_iterator begin() const override { return const_iterator(first, first == nullptr); }
	const_iterator end() const override { return const_iterator(last, true); }
	const_iterator cbegin() const { return const_iterator(first, first == nullptr); }
	const_iterator cend() const { return const_iterator(last, true); }

	/**
	 * Creates an empty LinkedList.
	 * Other effects: First, last, and cursor are all initialized to nullptr.  Always check list length before querying.
	 */
	LinkedList() {
		length = 0;
	}

	/**
	 * Loads the C array at src of size size.
	 * Time Complexity: O(size)
	 * Other effects: Cursor is set to the beginning of the list
	 * @param src The array to be loaded
	 * @param size The size of the array to be loaded
	 */
	LinkedList(const T *src, int size) {
		if (size > 0) {
			first = cursor = new linkedlist_value_container_t<T>(src[0]);
			first->previous = nullptr;
			if (size > 1) {
				for (int i = 1; i < size; i++) {
					cursor->next = new linkedlist_value_container_t<T>(src[i]);
					cursor->next->previous = cursor;
					cursor = cursor->next;
				}
				last = cursor;
				cursor = first;
			} else {
				last = first;
			}
			last->next = nullptr;
		}
		length = size;
	}

	/**
	 * Finds the container at the given index.  This is mostly for internal use.
	 * TIme complexity: O(i) (expected improvement in the future)
	 * @param i Index of item to find
	 * @return Container at index i, or nullptr if i is out of bounds
	 */
	linkedlist_value_container_t<T>* findItem(int i) const {
		if (i < 0 || i >= length)
			return nullptr;
		linkedlist_value_container_t<T>* current = first;
		for (int cnt = 0; cnt < i; cnt++) {
			current = current->next;
		}
		return current;
	}

	/**
	 * Adds an element to the endElem of the list
	 * Time complexity: O(1)
	 * Other effects: If the list is empty before calling this function, then the cursor will be set to the new element.
	 * @param item The item to add to the endElem of the list
	 */
	void add(T item) {
		length++;
		linkedlist_value_container_t<T>* newContainer = new linkedlist_value_container_t<T>(item);
		newContainer->previous = last;
		newContainer->next = nullptr;
		if (first) {
			last->next = newContainer;
			last = newContainer;
		} else
			first = last = cursor = newContainer;
	}

	void addMany(const T* items, int count) {
		for (int i = 0; i < count; i++)
			add(items[i]);
	}

	/**
	 * Adds an element to the start of the list
	 * Time complexity: O(1)
	 * Other effects: If the list is empty before calling this function, then the cursor will be set to the new element.
	 * @param item The item to add to the start of the list
	 */
	void insertAtBeginning(T item) {
		length++;
		linkedlist_value_container_t<T>* newContainer = new linkedlist_value_container_t<T>(item);
		newContainer->previous = nullptr;
		newContainer->next = first;
		if (last) {
			first->previous = newContainer;
			first = newContainer;
		} else
			first = last = cursor = newContainer;
	}

	/**
	 * Set the element at this location to the new element.
	 * Time complexity: O(i)
	 * Other effects: None
	 * @param i The index to set
	 * @param item The value to replace the previous value at i
	 */
	void set(int i, const T& item) {
		if (i < length)
			findItem(i)->value = item;
	}

	/**
	 * Remove the last element of the list
	 * Time complexity: O(1)
	 * Other effects: The cursor will be set to nullptr if it currently points to the last element
	 * @return The value removed from the endElem of the list
	 */
	T pop() {
		return removeRaw(last);
	}

	/**
	 * Removes the element at index i from the list, preserving order.
	 * Time complexity: O(i)
	 * Other affects: None; does not affect cursor position
	 * Returns: The removed element
	 */
	T remove(int i) {
		linkedlist_value_container_t<T>* containerToRemove = findItem(i);
		return removeRaw(containerToRemove);
	}

	/**
	 * Removes the given element, preserving order.  If the element is not present, then the list is unmodified.
	 * Time complexity: O(i)
	 * Other affects: None; does not affect cursor position
	 * Returns: True if the element was present, false otherwise
	 */
	bool removeByElement(T element) {
		linkedlist_value_container_t<T>* container = first;
		while (container) {
			if (container->value == element) {
				removeRaw(container);
				return true;
			}
			container = container->next;
		}
		return false;
	}

	/**
	 * Removes the cursor element, preserving order.
	 * Time complexity: O(1)
	 * Other affects: Cursor position becomes the next element
	 * Returns: The removed element
	 */
	T remove() {
		return removeRaw(cursor);
	}

	/**
	 * Sets the cursor to the first element, or nullptr if the list is empty
	 */
	void resetCursor() {
		cursor = first;
	}

	/**
	 * Moves the cursor to the last element
	 */
	void cursorToLast() {
		cursor = last;
	}

	/**
	 * Sets the cursor to this index in the list
	 * Time complexity: O(i)
	 * Other effects: None
	 * @param i
	 */
	void setCursor(int i) {
		cursor = findItem(i);
	}

	/**
	 * Get the value at the current cursor position
	 * Time complexity: O(1)
	 * Other effects: None
	 * @return Value of cursor
	 */
	T& getCursor() {
		return cursor->value;
	}

	/**
	 * Get the last value
	 * Time complexity: O(1)
	 * Other effects: None
	 * @return Value of the last element of the list
	 */
	T& getLast() {
		return last->value;
	}

	/**
	 * Returns the current cursor element and pushes the cursor forward one.  Will return nullptr if the endElem of the
	 * list is reached.
	 * @return Current cursor element, or nullptr if the cursor is exhausted.
	 */
	T& next() {
		T& value = cursor->value;
		cursor = cursor->next;
		return value;
	}

	bool isCursorValid() {
		return cursor != nullptr;
	}

	inline int size() const { return length; }

	T& get(int i) const {
		if (i >= length || i < 0)
			printf("Out of bounds read of %i for LinkedList of length %i\n", i, length);
		return findItem(i)->value;
	}

	~LinkedList() {
		clear();
	}

	/**
	 * Removes every element of the list
	 * Time complexity: O(size)
	 * Other effects: Cursor is set to nullptr
	 */
	inline void clear() {
		length = 0;
		cursor = first;
		while (cursor) {
			linkedlist_value_container_t<T>* toDelete = cursor;
			cursor = cursor->next;
			delete toDelete;
		}
		first = last = cursor = nullptr;
	}

	inline linkedlist_value_container_t<T>* getRawCursor() { return cursor; }
	inline void setRawCursor(linkedlist_value_container_t<T>* newCursor) { cursor = newCursor; }

public:
	int length;
	linkedlist_value_container_t<T>* first = nullptr;
	linkedlist_value_container_t<T>* last = nullptr;
	linkedlist_value_container_t<T>* cursor = nullptr;

private:
	T removeRaw(linkedlist_value_container_t<T>* containerToRemove) {
		if (containerToRemove == last)
			last = containerToRemove->previous;
		else if (containerToRemove->next)
			containerToRemove->next->previous = containerToRemove->previous;

		if (containerToRemove == first)
			first = containerToRemove->next;
		else if (containerToRemove->previous)
			containerToRemove->previous->next = containerToRemove->next;

		if (containerToRemove == cursor)
			cursor = containerToRemove->next;

		length--;

		T value = containerToRemove->value;
		delete containerToRemove;
		return value;
	}
};

#endif //EXCESSIVE_LINKEDLIST_H
