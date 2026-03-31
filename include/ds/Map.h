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


#ifndef LIBEXCESSIVE_MAP_H
#define LIBEXCESSIVE_MAP_H


#include "Container.h"


template <class K, class T>
class Map;


template <class K, class T>
struct MapElement {
	const K& key;
	T& value;

	class Iterator {
	public:
		using iterator_category = std::bidirectional_iterator_tag;
		using value_type = MapElement<K, T>;
		using difference_type = std::ptrdiff_t;
		using pointer = MapElement<K, T>*;
		using reference = MapElement<K, T>;

		Iterator(Map<K, T>* map, long index)
			: map(map), index(index) {}

		MapElement<K, T> operator*() {
			return map->getElementAtIndex(index);
		}

		Iterator& operator++() {
			try {
				map->getNext(index);
			} catch (const std::out_of_range&) {
				index = -1;
			}
			return *this;
		}

		Iterator operator++(int) {
			Iterator tmp = *this;
			++(*this);
			return tmp;
		}

		Iterator& operator--() {
			try {
				map->getPrevious(index);
			} catch (const std::out_of_range&) {
				index = -1;
			}
			return *this;
		}

		Iterator operator--(int) {
			Iterator tmp = *this;
			--(*this);
			return tmp;
		}

		bool operator==(const Iterator& other) const {
			return map == other.map && index == other.index;
		}

		bool operator!=(const Iterator& other) const {
			return !(*this == other);
		}

		Map<K, T>* map;
		long index;
	};

	class ConstIterator {
	public:
		using iterator_category = std::bidirectional_iterator_tag;
		using value_type = const MapElement<K, T>;
		using difference_type = std::ptrdiff_t;
		using pointer = const MapElement<K, T>*;
		using reference = const MapElement<K, T>;

		ConstIterator(const Map<K, T>* map, long index)
			: map(map), index(index) {}

		ConstIterator(const Iterator& other)
			: map(other.map), index(other.index) {}

		MapElement<K, T> operator*() const {
			return map->getElementAtIndex(index);
		}

		ConstIterator& operator++() {
			try {
				map->getNext(index);
			} catch (const std::out_of_range&) {
				index = -1;
			}
			return *this;
		}

		ConstIterator operator++(int) {
			ConstIterator tmp = *this;
			++(*this);
			return tmp;
		}

		ConstIterator& operator--() {
			try {
				map->getPrevious(index);
			} catch (const std::out_of_range&) {
				index = -1;
			}
			return *this;
		}

		ConstIterator operator--(int) {
			ConstIterator tmp = *this;
			--(*this);
			return tmp;
		}

		bool operator==(const ConstIterator& other) const {
			return map == other.map && index == other.index;
		}

		bool operator!=(const ConstIterator& other) const {
			return !(*this == other);
		}

		const Map<K, T>* map;
		long index;
	};
};


/**
 * @brief Interface for map-based containers.
 *
 * This class provides a standard interface for key-value collections. It inherits
 * from Container and uses MapElement as its element type.
 *
 * @tparam K The type of keys.
 * @tparam T The type of values.
 */
template <class K, class T>
class Map: public Container<MapElement<K, T>, MapElement<K, T>, typename MapElement<K, T>::Iterator, typename MapElement<K, T>::ConstIterator> {
public:
	/**
	 * @brief Retrieves the value associated with the given key.
	 * @param key The key to look for.
	 * @return The value associated with the key.
	 * @throws std::out_of_range if the key is not found.
	 */
	virtual T get(K key) const = 0;

	/**
	 * @brief Retrieves the value associated with the given key, or a default value if not found.
	 * @param key The key to look for.
	 * @param defaultValue The value to return if the key is not present.
	 * @return The associated value or defaultValue.
	 */
	virtual T getOrDefault(K key, T defaultValue) const = 0;

	/**
	 * @brief Retrieves a pointer to the value associated with the given key.
	 * @param key The key to look for.
	 * @return Pointer to the value, or nullptr if not found.
	 */
	virtual T* getPtr(K key) = 0;

	/**
	 * @brief Retrieves a constant pointer to the value associated with the given key.
	 * @param key The key to look for.
	 * @return Constant pointer to the value, or nullptr if not found.
	 */
	virtual const T* getPtr(K key) const = 0;

	/**
	 * @brief Checks if the map contains the specified key.
	 * @param key The key to check.
	 * @return true if the key is present, false otherwise.
	 */
	virtual bool hasKey(K key) const = 0;

	/**
	 * @brief Associates a value with a key using a pointer.
	 * @param key The key to associate with.
	 * @param value Pointer to the value.
	 * @return Reference to the newly associated value.
	 */
	virtual T& putPtr(K key, T* value) = 0;

	/**
	 * @brief Associates a value with a key.
	 * @param key The key to associate with.
	 * @param value The value to associate.
	 * @return Reference to the newly associated value.
	 */
	virtual T& put(K key, const T& value) = 0;

	/**
	 * @brief Associates a value with a key using move semantics.
	 * @param key The key to associate with.
	 * @param value The value to move.
	 * @return Reference to the newly associated value.
	 */
	virtual T& put(K key, T&& value) = 0;

	/**
	 * @brief Removes the entry with the specified key.
	 * @param key The key to remove.
	 * @return The value that was associated with the key.
	 * @throws std::out_of_range if the key is not found.
	 */
	virtual T remove(K key) = 0;

	/**
	 * @brief Removes the entry with the specified key and retrieves its value.
	 * @param key The key to remove.
	 * @param out [out] The value that was associated with the key.
	 * @return true if the key was found and removed, false otherwise.
	 */
	virtual bool remove(K key, T& out) = 0;

	/**
	 * @brief Gets the first element and its index.
	 * @param index [out] The index of the first element.
	 * @return The first MapElement.
	 */
	virtual MapElement<K, T> getFirst(long& index) = 0;

	/**
	 * @brief Gets the last element and its index.
	 * @param index [out] The index of the last element.
	 * @return The last MapElement.
	 */
	virtual MapElement<K, T> getLast(long& index) = 0;

	/**
	 * @brief Gets the next element and updates the index.
	 * @param index [in,out] The current index, updated to the next index.
	 * @return The next MapElement.
	 */
	virtual MapElement<K, T> getNext(long& index) = 0;

	/**
	 * @brief Gets the previous element and updates the index.
	 * @param index [in,out] The current index, updated to the previous index.
	 * @return The previous MapElement.
	 */
	virtual MapElement<K, T> getPrevious(long& index) = 0;

	/**
	 * @brief Gets the element at the specified internal index.
	 * @param index The internal index.
	 * @return The MapElement at that index.
	 */
	virtual MapElement<K, T> getElementAtIndex(long index) = 0;

	/**
	 * @brief Gets the first element and its index (const version).
	 * @param index [out] The index of the first element.
	 * @return The first MapElement.
	 */
	virtual const MapElement<K, T> getFirst(long& index) const = 0;

	/**
	 * @brief Gets the last element and its index (const version).
	 * @param index [out] The index of the last element.
	 * @return The last MapElement.
	 */
	virtual const MapElement<K, T> getLast(long& index) const = 0;

	/**
	 * @brief Gets the next element and updates the index (const version).
	 * @param index [in,out] The current index, updated to the next index.
	 * @return The next MapElement.
	 */
	virtual const MapElement<K, T> getNext(long& index) const = 0;

	/**
	 * @brief Gets the previous element and updates the index (const version).
	 * @param index [in,out] The current index, updated to the previous index.
	 * @return The previous MapElement.
	 */
	virtual const MapElement<K, T> getPrevious(long& index) const = 0;

	/**
	 * @brief Gets the element at the specified internal index (const version).
	 * @param index The internal index.
	 * @return The MapElement at that index.
	 */
	virtual const MapElement<K, T> getElementAtIndex(long index) const = 0;

	typename MapElement<K, T>::Iterator begin() override {
		long index = -1;
		if (!this->isEmpty())
			getFirst(index);
		return typename MapElement<K, T>::Iterator(this, index);
	}

	typename MapElement<K, T>::Iterator end() override {
		return typename MapElement<K, T>::Iterator(this, -1);
	}

	typename MapElement<K, T>::ConstIterator begin() const override {
		long index = -1;
		if (!this->isEmpty())
			getFirst(index);
		return typename MapElement<K, T>::ConstIterator(this, index);
	}

	typename MapElement<K, T>::ConstIterator end() const override {
		return typename MapElement<K, T>::ConstIterator(this, -1);
	}

	virtual void clear() override = 0;

	virtual ~Map() = default;
};

#endif //LIBEXCESSIVE_MAP_H
