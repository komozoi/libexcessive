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


template <class K, class T>
class Map: public Container<MapElement<K, T>, MapElement<K, T>, typename MapElement<K, T>::Iterator, typename MapElement<K, T>::ConstIterator> {
public:
	virtual T get(K key) const = 0;
	virtual T getOrDefault(K key, T defaultValue) const = 0;
	virtual T* getPtr(K key) = 0;
	virtual const T* getPtr(K key) const = 0;

	virtual bool hasKey(K key) const = 0;

	virtual T& putPtr(K key, T* value) = 0;
	virtual T& put(K key, const T& value) = 0;
	virtual T& put(K key, T&& value) = 0;

	virtual T remove(K key) = 0;
	virtual bool remove(K key, T& out) = 0;

	// Gets the first/last and assigns the index
	virtual MapElement<K, T> getFirst(long& index) = 0;
	virtual MapElement<K, T> getLast(long& index) = 0;
	virtual MapElement<K, T> getNext(long& index) = 0;
	virtual MapElement<K, T> getPrevious(long& index) = 0;
	virtual MapElement<K, T> getElementAtIndex(long index) = 0;
	virtual const MapElement<K, T> getFirst(long& index) const = 0;
	virtual const MapElement<K, T> getLast(long& index) const = 0;
	virtual const MapElement<K, T> getNext(long& index) const = 0;
	virtual const MapElement<K, T> getPrevious(long& index) const = 0;
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

	virtual void clear() = 0;

	virtual ~Map() = default;
};

#endif //LIBEXCESSIVE_MAP_H
