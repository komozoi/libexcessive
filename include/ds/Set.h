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
	virtual bool add(const T& item) = 0;

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
	virtual bool contains(const T& query) const = 0;

	/**
	 * @brief Removes the specified element from the set.
	 * @param key The element to remove.
	 * @return true if the element was found and removed, false otherwise.
	 */
	virtual bool remove(const T& key) = 0;

	/**
	 * @brief Checks whether this set is a subset of another set.
	 *
	 * Every element of this set must be present in `other`. Works across
	 * different Set specializations (e.g. HashSet vs ArraySet).
	 *
	 * @tparam OtherIterator Iterator type of the other set.
	 * @tparam OtherConstIterator Const iterator type of the other set.
	 * @param other The set to compare against.
	 * @return true if this set is a subset of other (including when equal).
	 */
	template<class OtherIterator, class OtherConstIterator>
	bool isSubsetOf(const Set<T, OtherIterator, OtherConstIterator>& other) const {
		if (this->size() > other.size())
			return false;
		for (ConstIterator it = this->begin(); it != this->end(); ++it) {
			if (!other.contains(*it))
				return false;
		}
		return true;
	}

	/**
	 * @brief Checks whether this set is a superset of another set.
	 *
	 * Every element of `other` must be present in this set. Works across
	 * different Set specializations (e.g. HashSet vs ArraySet).
	 *
	 * @tparam OtherIterator Iterator type of the other set.
	 * @tparam OtherConstIterator Const iterator type of the other set.
	 * @param other The set to compare against.
	 * @return true if this set is a superset of other (including when equal).
	 */
	template<class OtherIterator, class OtherConstIterator>
	bool isSupersetOf(const Set<T, OtherIterator, OtherConstIterator>& other) const {
		if (this->size() < other.size())
			return false;
		for (OtherConstIterator it = other.begin(); it != other.end(); ++it) {
			if (!this->contains(*it))
				return false;
		}
		return true;
	}

	virtual ~Set() = default;
};


/**
 * @brief Equality comparison for sets of the same element type.
 *
 * Free function (not a member) so it is only instantiated when used and so
 * Set specializations with different iterators remain comparable without
 * requiring a single shared Set base type. Two sets are equal when they
 * contain exactly the same elements (order is irrelevant).
 *
 * Element equality is determined via each set's `contains` implementation
 * rather than a direct `T::operator==` requirement on this function itself.
 *
 * @return true if both sets have the same size and the same elements.
 */
template<class T, class It1, class CIt1, class It2, class CIt2>
bool operator==(const Set<T, It1, CIt1>& a, const Set<T, It2, CIt2>& b) {
	return a.size() == b.size() && a.isSubsetOf(b);
}

/**
 * @brief Inequality comparison for sets of the same element type.
 * @return true if the sets do not contain exactly the same elements.
 */
template<class T, class It1, class CIt1, class It2, class CIt2>
bool operator!=(const Set<T, It1, CIt1>& a, const Set<T, It2, CIt2>& b) {
	return !(a == b);
}


#endif //LIBEXCESSIVE_SET_H
