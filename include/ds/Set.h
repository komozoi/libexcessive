/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-02-24
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


#ifndef LIBEXCESSIVE_SET_H
#define LIBEXCESSIVE_SET_H

#include "Container.h"


/**
 * @brief Interface for set-based containers.
 *
 * This class provides a standard interface for collections of unique elements.
 * It inherits from Container.
 *
 * @tparam T The type of elements in the set.
 * @tparam Iterator The iterator type.
 * @tparam ConstIterator The constant iterator type.
 */
template<class T, class Iterator = T*, class ConstIterator = const T*>
class Set: public Container<T, T&, Iterator, ConstIterator> {
public:
	/**
	 * @brief Adds an item to the set.
	 * @param item Item to add.
	 * @return `true` if item was already present, `false` otherwise.
	 */
	virtual bool add(T item) = 0;

	/**
	 * @brief Adds multiple items from a raw array.
	 * @param values Pointer to the source array.
	 * @param count Number of elements to add.
	 */
	virtual void addMany(const T* values, int count) {
		for (int i = 0; i < count; i++)
			add(values[i]);
	}

	/**
	 * @brief Checks if the set contains the specified element.
	 * @param query The element to check.
	 * @return true if the element is present, false otherwise.
	 */
	virtual bool contains(T query) const = 0;

	/**
	 * @brief Removes the specified element from the set.
	 * @param key The element to remove.
	 * @return true if the element was found and removed, false otherwise.
	 */
	virtual bool remove(T key) = 0;

	virtual void clear() override = 0;

	virtual ~Set() = default;
};


#endif //LIBEXCESSIVE_SET_H
