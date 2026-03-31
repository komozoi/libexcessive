/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-08-15
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


#ifndef EXCESSIVE_SHORTQUEUE_H
#define EXCESSIVE_SHORTQUEUE_H

#include "stdlib.h"


/**
 * @brief A priority queue that maintains a sorted array of limited size.
 *
 * This class keeps elements sorted by priority (lowest to highest). If the queue
 * is full, adding a new element with a lower priority than the current highest
 * priority will replace the highest priority element.
 *
 * @tparam T The type of values.
 * @tparam P The type of priorities.
 */
template <class T, typename P>
class ShortQueue {
public:
	/**
	 * @brief Constructs a ShortQueue with the specified capacity.
	 * @param count Maximum number of elements.
	 */
	explicit ShortQueue(int count) : count(count), numUsed(0) {
		elements = (element_t*)malloc(sizeof(element_t) * count);
	}

	/**
	 * @brief Inserts an element with the given priority.
	 * @param value The value to insert.
	 * @param priority The priority associated with the value.
	 */
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

	/**
	 * @brief Returns a constant reference to the value at the specified index.
	 * @param i Index.
	 * @return Reference to the value.
	 */
	const T& at(int i) const {
		return elements[i].value;
	}

	/**
	 * @brief Updates the priority of an element at the specified index.
	 * @param i Index.
	 * @param priority New priority.
	 */
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

	/**
	 * @brief Returns a reference to the element with the minimum priority.
	 * @return Reference to the minimum value.
	 */
	T& getMin() {
		return elements[0].value;
	}

	/**
	 * @brief Returns the minimum priority value.
	 * @return The minimum priority.
	 */
	const P& minPriority() const {
		return elements[0].priority;
	}

	/**
	 * @brief Returns a constant reference to the element with the minimum priority.
	 * @return Constant reference to the minimum value.
	 */
	const T& getMin() const {
		return elements[0].value;
	}

	/**
	 * @brief Removes and returns the element with the minimum priority.
	 * @return The minimum value.
	 */
	T deleteMin() {
		if (numUsed == 0)
			return T();

		T out = elements[0].value;

		numUsed--;
		for (int i = 0; i < numUsed; i++)
			elements[i] = std::move(elements[i + 1]);

		return out;
	}

	/**
	 * @brief Removes the minimum element and retrieves its value and priority.
	 * @param valOut [out] The minimum value.
	 * @param priorityOut [out] The minimum priority.
	 * @return true if an element was removed, false if the queue was empty.
	 */
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

	/**
	 * @brief Checks if the queue is empty.
	 * @return true if empty, false otherwise.
	 */
	bool isEmpty() const {
		return numUsed == 0;
	}

	/**
	 * @brief Returns the number of elements in the queue.
	 * @return The current size.
	 */
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
