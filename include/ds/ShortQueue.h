//
// Created by komozoi on 15.8.25.
//

#ifndef EXCESSIVE_SHORTQUEUE_H
#define EXCESSIVE_SHORTQUEUE_H

#include "stdlib.h"


template <class T, typename P>
class ShortQueue {
public:
	explicit ShortQueue(int count) : count(count), numUsed(0) {
		elements = (element_t*)malloc(sizeof(element_t) * count);
	}

	void insert(const T& value, const P& priority) {
		// If full, we may need to remove the highest priority item
		if (numUsed == count) {
			// If the new priority is lower (better) than the highest priority (last element), insert
			if (priority < elements[numUsed - 1].priority) {
				// Remove the last (highest priority) element
				numUsed--;
			} else {
				// Do nothing because the new item is worse than any in the queue
				return;
			}
		}

		// Find the insert position (keep sorted from lowest to highest priority)
		int i = numUsed - 1;
		while (i >= 0 && elements[i].priority > priority) {
			elements[i + 1] = std::move(elements[i]);  // Shift right
			--i;
		}

		// Insert the new element
		elements[i + 1].value = value;
		elements[i + 1].priority = priority;

		++numUsed;
	}

	const T& at(int i) const {
		return elements[i].value;
	}

	void updatePriorityAt(int i, const P& priority) {
		if (i < 0 || i >= numUsed) {
			// Invalid index; silently ignore or assert
			return;
		}

		// Save the element to be updated
		element_t updated = elements[i];
		updated.priority = priority;

		// Shift elements left to remove index i
		for (int j = i; j < numUsed - 1; ++j) {
			elements[j] = std::move(elements[j + 1]);
		}
		--numUsed;

		// Reinsert the updated element in the correct position
		int pos = numUsed - 1;
		while (pos >= 0 && elements[pos].priority > updated.priority) {
			elements[pos + 1] = std::move(elements[pos]);  // Shift right
			--pos;
		}

		elements[pos + 1] = std::move(updated);
		++numUsed;
	}

	T& getMin() {
		return elements[0].value;
	}

	const P& minPriority() const {
		return elements[0].priority;
	}

	const T& getMin() const {
		return elements[0].value;
	}

	T deleteMin() {
		if (numUsed == 0)
			return T();

		T out = elements[0].value;

		numUsed--;
		for (int i = 0; i < numUsed; i++)
			elements[i] = std::move(elements[i + 1]);

		return out;
	}

	bool deleteMin(T* valOut, P* priorityOut) {
		if (numUsed == 0)
			return false;

		element_t& tmp = elements[0];
		*valOut = tmp.value;
		*priorityOut = tmp.priority;

		numUsed--;
		for (int i = 0; i < numUsed; i++)
			elements[i] = std::move(elements[i + 1]);

		return true;
	}

	bool isEmpty() const {
		return numUsed == 0;
	}

	int size() const {
		return numUsed;
	}

	~ShortQueue() {
		for (int i = 0; i < numUsed; i++)
			elements[i].~element_t();
		free(elements);
	}


private:
	struct element_t {
		T value;
		P priority;
	};

	int count, numUsed;
	element_t* elements;
};

#endif //EXCESSIVE_SHORTQUEUE_H
