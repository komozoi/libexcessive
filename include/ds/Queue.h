//
// Created by nuclaer on 19.2.25.
//

#ifndef EXCESSIVE_QUEUE_H
#define EXCESSIVE_QUEUE_H

#include "alloc/SlabAllocator.h"


template <class T>
class Queue {

	class QueueElement {
	public:
		QueueElement(T value, QueueElement* previous, QueueElement* next)
			: value(value), previous(previous), next(next) {}

		T value;
		QueueElement* previous;
		QueueElement* next;
	};

public:

	Queue() {}

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

	explicit Queue(const T& value) {
		add(value);
	}

	explicit Queue(T&& value) {
		add(value);
	}

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

	void add(const T& value) {
		QueueElement* newLast = allocator.allocate(QueueElement(value, last, nullptr));
		if (first == nullptr)
			first = newLast;
		else
			last->next = newLast;
		last = newLast;
		n++;
	}

	void add(T&& value) {
		QueueElement* newLast = allocator.allocate(QueueElement(value, last, nullptr));
		if (first == nullptr)
			first = newLast;
		else
			last->next = newLast;
		last = newLast;
		n++;
	}

	void setLast(const T& value) {
		last->value = value;
	}

	void setLast(T&& value) {
		last->value = value;
	}

	bool empty() {
		return first == nullptr;
	}

	int size() {
		return n;
	}

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

	T peek() const {
		return first->value;
	}

	T peekLast() const {
		return last->value;
	}

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
