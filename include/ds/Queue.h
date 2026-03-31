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

#include "alloc/SlabAllocator.h"


/**
 * @brief A linked-list based queue implementation.
 *
 * This class provides a FIFO collection of elements using a doubly-linked
 * list internally. It uses a SlabAllocator for efficient memory management.
 *
 * @tparam T The type of elements.
 */
template <class T>
class Queue {
	/**
	 * @brief Internal node structure for the queue.
	 */
	class QueueElement {
	public:
		/**
		 * @brief Constructs a queue element.
		 * @param value The value to store.
		 * @param previous Pointer to the previous element.
		 * @param next Pointer to the next element.
		 */
		QueueElement(T value, QueueElement* previous, QueueElement* next)
			: value(value), previous(previous), next(next) {}

		T value;
		QueueElement* previous;
		QueueElement* next;
	};

public:

	/**
	 * @brief Constructs an empty queue.
	 */
	Queue() {}

	/**
	 * @brief Copy constructor.
	 * @param other The queue to copy from.
	 */
	Queue(const Queue& other) {
		QueueElement* element = other.first;
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
			slab.freeAll();
			n = 0;
			first = nullptr;
			last = nullptr;

			QueueElement *element = rhs.first;
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
		QueueElement* newLast = allocator.allocate(QueueElement(value, last, nullptr));
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
		QueueElement* newLast = allocator.allocate(QueueElement(value, last, nullptr));
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
	bool empty() {
		return first == nullptr;
	}

	/**
	 * @brief Returns the number of elements in the queue.
	 * @return The size.
	 */
	int size() {
		return n;
	}

	/**
	 * @brief Removes and returns the first element in the queue.
	 * @return The first element.
	 */
	T pop() {
		T tmp = std::move(first->value);
		QueueElement* oldFirst = first;
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
	void clear() {
		QueueElement* elem = first;
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
	}

	~Queue() {
		clear();
	}

private:
	QueueElement* first = nullptr;
	QueueElement* last = nullptr;

	int n = 0;

	// This just makes the datastructure faster + more memory efficient, as long as the queue is large enough.
	// For small queues the burden is low.
	SlabAllocator allocator;
};


#endif //EXCESSIVE_QUEUE_H
