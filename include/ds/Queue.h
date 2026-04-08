/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-02-19
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


#ifndef EXCESSIVE_QUEUE_H
#define EXCESSIVE_QUEUE_H

#include "Container.h"
#include "alloc/SlabAllocator.h"
#include <iterator>

/**
 * @brief Internal node structure for the queue.
 */
template <typename T>
struct excessive_queue_element_t {
	excessive_queue_element_t(const T& value, excessive_queue_element_t* previous, excessive_queue_element_t* next)
	: previous(previous), next(next), value(value) {}

	excessive_queue_element_t(T&& value, excessive_queue_element_t* previous, excessive_queue_element_t* next)
		: previous(previous), next(next), value(value) {}

	excessive_queue_element_t* previous;
	excessive_queue_element_t* next;
	T value;
};

/**
 * @brief Iterator class for the queue.
 */
template <typename T>
class QueueIterator {
public:

	using iterator_category = std::bidirectional_iterator_tag;
	using value_type = T;
	using difference_type = std::ptrdiff_t;
	using pointer = T*;
	using reference = T&;

	QueueIterator(excessive_queue_element_t<T>* element) : current(element) {}

	T& operator*() const { return current->value; }
	T* operator->() { return &current->value; }

	QueueIterator& operator++() {
		if (current) current = current->next;
		return *this;
	}

	QueueIterator operator++(int) {
		QueueIterator tmp = *this;
		++(*this);
		return tmp;
	}

	QueueIterator& operator--() {
		if (current) current = current->previous;
		return *this;
	}

	QueueIterator operator--(int) {
		QueueIterator tmp = *this;
		--(*this);
		return tmp;
	}

	bool operator==(const QueueIterator& other) const { return current == other.current; }
	bool operator!=(const QueueIterator& other) const { return current != other.current; }

private:
	excessive_queue_element_t<T>* current;
};

/**
 * @brief Constant iterator class for the queue.
 */
template <typename T>
class QueueConstIterator {
public:
	using iterator_category = std::bidirectional_iterator_tag;
	using value_type = const T;
	using difference_type = std::ptrdiff_t;
	using pointer = const T*;
	using reference = const T&;

	QueueConstIterator(const excessive_queue_element_t<T>* element) : current(element) {}

	const T& operator*() const { return current->value; }
	const T* operator->() const { return &current->value; }

	QueueConstIterator& operator++() {
		if (current) current = current->next;
		return *this;
	}

	QueueConstIterator operator++(int) {
		QueueConstIterator tmp = *this;
		++(*this);
		return tmp;
	}

	QueueConstIterator& operator--() {
		if (current) current = current->previous;
		return *this;
	}

	QueueConstIterator operator--(int) {
		QueueConstIterator tmp = *this;
		--(*this);
		return tmp;
	}

	bool operator==(const QueueConstIterator& other) const { return current == other.current; }
	bool operator!=(const QueueConstIterator& other) const { return current != other.current; }

private:
	const excessive_queue_element_t<T>* current;
};

/**
 * @brief A linked-list based queue implementation.
 *
 * This class provides a FIFO collection of elements using a doubly-linked
 * list internally. It uses a SlabAllocator for efficient memory management.
 *
 * @tparam T The type of elements.
 */
template <class T>
class Queue : public Container<T, T&, QueueIterator<T>, QueueConstIterator<T>> {
public:

	using Iterator = QueueIterator<T>;
	using ConstIterator = QueueConstIterator<T>;

	Iterator begin() override { return Iterator(first); }
	Iterator end() override { return Iterator(nullptr); }
	ConstIterator begin() const override { return ConstIterator(first); }
	ConstIterator end() const override { return ConstIterator(nullptr); }

	/**
	 * @brief Constructs an empty queue.
	 */
	Queue() {}

	/**
	 * @brief Copy constructor.
	 * @param other The queue to copy from.
	 */
	Queue(const Queue& other) {
		excessive_queue_element_t<T>* element = other.first;
		while (element) {
			add(element->value);
			element = element->next;
		}
	}

	// Currently we cannot move() a SlabAllocator safely.
	/*Queue(Queue&& other) noexcept : allocator(std::move(other.allocator)) {
		first = other.first;
		last = other.last;
		n = other.n;
		other.first = nullptr;
		other.last = nullptr;
		other.n = 0;
	}*/

	/**
	 * @brief Constructs a queue with one element.
	 * @param value The initial element.
	 */
	explicit Queue(const T& value) {
		add(value);
	}

	/**
	 * @brief Constructs a queue with one element using move semantics.
	 * @param value The initial element to move.
	 */
	explicit Queue(T&& value) {
		add(value);
	}

	/**
	 * @brief Copy assignment operator.
	 * @param rhs The queue to copy from.
	 * @return Reference to this queue.
	 */
	inline Queue& operator=(const Queue& rhs) {
		if (&rhs != this) {
			allocator.freeAll();
			n = 0;
			first = nullptr;
			last = nullptr;

			excessive_queue_element_t<T>* element = rhs.first;
			while (element) {
				add(element->value);
				element = element->next;
			}
		}

		return *this;
	}

	/**
	 * @brief Adds an element to the end of the queue.
	 * @param value The element to add.
	 */
	void add(const T& value) {
		excessive_queue_element_t<T>* newLast = allocator.allocate(excessive_queue_element_t<T>(value, last, nullptr));
		if (first == nullptr)
			first = newLast;
		else
			last->next = newLast;
		last = newLast;
		n++;
	}

	/**
	 * @brief Adds an element to the end of the queue using move semantics.
	 * @param value The element to move into the queue.
	 */
	void add(T&& value) {
		excessive_queue_element_t<T>* newLast = allocator.allocate(excessive_queue_element_t<T>(value, last, nullptr));
		if (first == nullptr)
			first = newLast;
		else
			last->next = newLast;
		last = newLast;
		n++;
	}

	/**
	 * @brief Updates the value of the last element.
	 * @param value New value for the last element.
	 */
	void setLast(const T& value) {
		last->value = value;
	}

	/**
	 * @brief Updates the value of the last element using move semantics.
	 * @param value New value for the last element.
	 */
	void setLast(T&& value) {
		last->value = value;
	}

	/**
	 * @brief Checks if the queue is empty.
	 * @return true if empty, false otherwise.
	 */
	bool empty() const {
		return first == nullptr;
	}

	/**
	 * @brief Checks if the queue is empty.
	 * @return true if empty, false otherwise.
	 */
	bool isEmpty() const override {
		return first == nullptr;
	}

	/**
	 * @brief Returns the number of elements in the queue.
	 * @return The size.
	 */
	int size() const override {
		return n;
	}

	/**
	 * @brief Removes and returns the first element in the queue.
	 * @return The first element.
	 */
	T pop() {
		T tmp = std::move(first->value);
		excessive_queue_element_t<T>* oldFirst = first;
		first = first->next;
		if (first)
			first->previous = nullptr;
		else
			last = nullptr;
		allocator.free(oldFirst);
		n--;
		return tmp;
	}

	/**
	 * @brief Returns the value of the first element without removing it.
	 * @return The first element.
	 */
	T peek() const {
		return first->value;
	}

	/**
	 * @brief Returns the value of the last element without removing it.
	 * @return The last element.
	 */
	T peekLast() const {
		return last->value;
	}

	/**
	 * @brief Removes all elements from the queue.
	 */
	void clear() override {
		excessive_queue_element_t<T>* elem = first;
		while (elem) {
			elem->value.~T();
			elem = elem->next;
		}

		// Although it will do this on its own, the deconstructor throws warnings if it
		// thinks there are memory leaks.  We do this so the slab knows that the memory
		// is no longer needed, and that deleting all of it is intentional.
		allocator.freeAll();

		first = nullptr;
		last = nullptr;
		n = 0;
	}

	~Queue() {
		clear();
	}

private:
	excessive_queue_element_t<T>* first = nullptr;
	excessive_queue_element_t<T>* last = nullptr;

	int n = 0;

	// This just makes the datastructure faster + more memory efficient, as long as the queue is large enough.
	// For small queues the burden is low.
	SlabAllocator allocator;
};


#endif //EXCESSIVE_QUEUE_H
