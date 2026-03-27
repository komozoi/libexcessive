/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2023-06-02
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


#ifndef EXCESSIVE_PRIORITYQUEUE_H
#define EXCESSIVE_PRIORITYQUEUE_H

#include <climits>

template <typename T, int capacity>
class StaticPriorityQueue {
public:

	/**
	 * Adds an item to the Queue.  If there is no space, and the last item in the queue has priority, then the new
	 * item is not added.  If there is no space, and the new item has priority over the last item, then the new item
	 * replaces the last item in the queue.
	 * @param item Item to be added
	 * @param priority Priority of new item
	 */
	void add(const T& item, int priority) {
		if (size == capacity) {
			if (priority < lowestPriority)
				return;
			int last = lastIndex();
			if (last > priority)
				return;
			content[last] = item;
			priorities[last] = priority;
			recomputeLowestPriority(-1);
		} else {
			content[size++] = item;
			if (priority < lowestPriority)
				lowestPriority = priority;
		}
	}

	/**
	 * @return The highest priority item in the queue
	 */
	T& first() {
		return content[firstIndex()];
	}

	/**
	 * @return The lowest priority item in the queue
	 */
	T& last() {
		return content[lastIndex()];
	}

	/**
	 * Removes and returns the first item in the queue
	 * @return the item that was first in the queue
	 */
	T& read() {
		return remove(firstIndex());
	}

	/**
	 * Removes and returns the last item in the queue
	 * @return the item that was last in the queue
	 */
	T& pop() {
		return remove(lastIndex());
	}

	T& remove(int i) {
		T& old = content[i];
		if (priorities[i] == lowestPriority) {
			recomputeLowestPriority(i);
		}
		content[i] = content[--size];
		priorities[i] = priorities[size];
		return old;
	}

	int lastIndex() {
		if (size == 0)
			return -1;
		int lowest = INT_MIN;
		int last = 0;
		for (int i = 0; i < size; i++) {
			if (lowestPriority == priorities[i])
				return i;
			if (priorities[i] < lowest) {
				lowest = priorities[i];
				last = i;
			}
		}
		return last;
	}

	int firstIndex() {
		if (size == 0)
			return -1;
		int highest = INT_MIN;
		int first = 0;
		for (int i = 0; i < size; i++)
			if (priorities[i] > highest) {
				highest = priorities[i];
				first = i;
			}
		return first;
	}

	T& at(int i) {
		return content[i];
	}

	int priorityAt(int i) {
		return priorities[i];
	}

	int length() {
		return size;
	}

	void sort() {
		for (int i = 1; i < size; i++) {
			for (int j = 0; j < i; j++) {
				if (priorities[i] > priorities[j]) {
					int tmpPriority = priorities[i];
					priorities[i] = priorities[j];
					priorities[j] = tmpPriority;

					T tmpValue = content[i];
					content[i] = content[j];
					content[j] = tmpValue;
				}
			}
		}
	}

	int highestPriority() {
		if (size == 0)
			return INT_MIN;
		return priorityAt(firstIndex());
	}

	int staticCapacity() {
		return capacity;
	}

	void setPriority(int i, int priority) {
		if (i >= size)
			return;
		if (size == 0 || priority < lowestPriority)
			lowestPriority = priority;
		priorities[i] = priority;
	}

	void clear() {
		size = 0;
		lowestPriority = INT_MAX;
	}

private:

	void recomputeLowestPriority(int ignore) {
		int lowest = INT_MAX;
		for (int i = 0; i < size; i++) {
			if (i != ignore) {
				if (priorities[i] < lowest)
					lowest = priorities[i];
			}
		}
		lowestPriority = lowest;
	}

	T content[capacity];
	int priorities[capacity]{0};
	int size = 0;
	int lowestPriority = INT_MAX;
};


#endif //EXCESSIVE_PRIORITYQUEUE_H
