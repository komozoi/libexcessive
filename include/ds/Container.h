/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2024-03-30
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

#ifndef LIBEXCESSIVE_CONTAINER_H
#define LIBEXCESSIVE_CONTAINER_H

#include <iterator>


/**
 * @brief Base interface for all container classes.
 *
 * This class provides a standard interface for collections, including iteration support
 * and common utility methods like size() and isEmpty().
 *
 * @tparam T The type of elements stored in the container.
 * @tparam R The return type for getElement (often T or T&).
 * @tparam Iterator The iterator type for the container.
 * @tparam ConstIterator The constant iterator type for the container.
 */
template <typename T, typename R, typename Iterator, typename ConstIterator>
class Container {
public:
	/**
	 * @brief Returns an iterator to the beginning of the container.
	 * @return Iterator to the first element.
	 */
	virtual Iterator begin() = 0;

	/**
	 * @brief Returns an iterator to the end of the container.
	 * @return Iterator to the element following the last element.
	 */
	virtual Iterator end() = 0;

	/**
	 * @brief Returns a constant iterator to the beginning of the container.
	 * @return ConstIterator to the first element.
	 */
	virtual ConstIterator begin() const = 0;

	/**
	 * @brief Returns a constant iterator to the end of the container.
	 * @return ConstIterator to the element following the last element.
	 */
	virtual ConstIterator end() const = 0;

	/**
	 * @brief Returns a constant iterator to the beginning of the container.
	 * @return ConstIterator to the first element.
	 */
	ConstIterator cbegin() const { return begin(); }

	/**
	 * @brief Returns a constant iterator to the end of the container.
	 * @return ConstIterator to the element following the last element.
	 */
	ConstIterator cend() const { return end(); }

	typedef std::reverse_iterator<Iterator> reverse_iterator;
	typedef std::reverse_iterator<ConstIterator> const_reverse_iterator;

	/**
	 * @brief Returns a reverse iterator to the first element of the reversed container.
	 * @return reverse_iterator to the last element.
	 */
	reverse_iterator rbegin() { return reverse_iterator(end()); }

	/**
	 * @brief Returns a reverse iterator to the element following the last element of the reversed container.
	 * @return reverse_iterator to the element before the first element.
	 */
	reverse_iterator rend() { return reverse_iterator(begin()); }

	/**
	 * @brief Returns a constant reverse iterator to the first element of the reversed container.
	 * @return const_reverse_iterator to the last element.
	 */
	const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }

	/**
	 * @brief Returns a constant reverse iterator to the element following the last element of the reversed container.
	 * @return const_reverse_iterator to the element before the first element.
	 */
	const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

	/**
	 * @brief Returns a constant reverse iterator to the first element of the reversed container.
	 * @return const_reverse_iterator to the last element.
	 */
	const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }

	/**
	 * @brief Returns a constant reverse iterator to the element following the last element of the reversed container.
	 * @return const_reverse_iterator to the element before the first element.
	 */
	const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

	/**
	 * @brief Returns the number of elements in the container.
	 * @return Number of elements.
	 */
	virtual int size() const = 0;

	/**
	 * @brief Checks if the container is empty.
	 * @return true if the container contains no elements, false otherwise.
	 */
	virtual bool isEmpty() const { return size() == 0; }

	/**
	 * @brief Gets the element at the specified index.
	 * @param i The zero-based index of the element to retrieve.
	 * @return The element at the specified index.
	 * @throws std::out_of_range if the index is out of bounds.
	 */
	virtual R getElement(int i) {
		int count = 0;
		for (auto it = begin(); it != end(); ++it) {
			if (count == i) {
				return *it;
			}
			count++;
		}
		throw std::out_of_range("Index out of bounds");
	}

	/**
	 * @brief Finds the index of the first occurrence of the specified item.
	 * @param item The item to search for.
	 * @return The zero-based index of the first occurrence, or -1 if not found.
	 */
	virtual int find(const T& item) {
		int index = 0;
		for (auto it = begin(); it != end(); ++it) {
			if (compare(*it, item)) {
				return index;
			}
			index++;
		}
		return -1;
	}

	/**
	 * @brief Removes all elements from the container.
	 */
	virtual void clear() = 0;

	virtual ~Container() = default;

private:
	template <typename U>
	auto compare(const U& a, const U& b) const -> decltype(a == b, bool()) {
		return a == b;
	}

	bool compare(...) const {
		return false;
	}
};

#endif //LIBEXCESSIVE_CONTAINER_H