//
// Created by komozoi on 8/1/23.
//

#ifndef EXCESSIVE_LINKEDLIST_H
#define EXCESSIVE_LINKEDLIST_H


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iterator>
#include <cstddef>


template<class T>
class linkedlist_value_container_t {
public:
	explicit inline linkedlist_value_container_t(const T& value) : value(value) {}
	T value;
	linkedlist_value_container_t* next;
	linkedlist_value_container_t* previous;
};


template<class T>
class LinkedList {
public:

	class iterator {
	public:
		typedef std::bidirectional_iterator_tag iterator_category;
		typedef T value_type;
		typedef std::ptrdiff_t difference_type;
		typedef T* pointer;
		typedef T& reference;

		iterator() : node(nullptr), list(nullptr) {}
		iterator(linkedlist_value_container_t<T>* node, LinkedList<T>* list) : node(node), list(list) {}

		reference operator*() const { return node->value; }
		pointer operator->() const { return &node->value; }

		iterator& operator++() {
			if (node) node = node->next;
			return *this;
		}

		iterator operator++(int) {
			iterator tmp = *this;
			++(*this);
			return tmp;
		}

		iterator& operator--() {
			node = node ? node->previous : (list ? list->last : nullptr);
			return *this;
		}

		iterator operator--(int) {
			iterator tmp = *this;
			--(*this);
			return tmp;
		}

		bool operator==(const iterator& other) const { return node == other.node; }
		bool operator!=(const iterator& other) const { return node != other.node; }

	private:
		linkedlist_value_container_t<T>* node;
		LinkedList<T>* list;
		friend class LinkedList;
		friend class const_iterator;
	};

	class const_iterator {
	public:
		typedef std::bidirectional_iterator_tag iterator_category;
		typedef const T value_type;
		typedef std::ptrdiff_t difference_type;
		typedef const T* pointer;
		typedef const T& reference;

		const_iterator() : node(nullptr), list(nullptr) {}
		const_iterator(const linkedlist_value_container_t<T>* node, const LinkedList<T>* list) : node(node), list(list) {}
		const_iterator(const iterator& other) : node(other.node), list(other.list) {}

		reference operator*() const { return node->value; }
		pointer operator->() const { return &node->value; }

		const_iterator& operator++() {
			if (node) node = node->next;
			return *this;
		}

		const_iterator operator++(int) {
			const_iterator tmp = *this;
			++(*this);
			return tmp;
		}

		const_iterator& operator--() {
			node = node ? node->previous : (list ? list->last : nullptr);
			return *this;
		}

		const_iterator operator--(int) {
			const_iterator tmp = *this;
			--(*this);
			return tmp;
		}

		bool operator==(const const_iterator& other) const { return node == other.node; }
		bool operator!=(const const_iterator& other) const { return node != other.node; }

	private:
		const linkedlist_value_container_t<T>* node;
		const LinkedList<T>* list;
		friend class LinkedList;
	};

	iterator begin() { return iterator(first, this); }
	iterator end() { return iterator(nullptr, this); }
	const_iterator begin() const { return const_iterator(first, this); }
	const_iterator end() const { return const_iterator(nullptr, this); }
	const_iterator cbegin() const { return const_iterator(first, this); }
	const_iterator cend() const { return const_iterator(nullptr, this); }

	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

	reverse_iterator rbegin() { return reverse_iterator(end()); }
	reverse_iterator rend() { return reverse_iterator(begin()); }
	const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
	const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
	const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
	const_reverse_iterator crend() const { return const_reverse_iterator(begin()); }

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

	int length;
	linkedlist_value_container_t<T>* first = nullptr;
	linkedlist_value_container_t<T>* last = nullptr;
	linkedlist_value_container_t<T>* cursor = nullptr;
};

#endif //EXCESSIVE_LINKEDLIST_H
