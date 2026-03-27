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

template <class K, class T>
class Map {
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

	virtual void clear() = 0;

	virtual ~Map() = default;
};

#endif //LIBEXCESSIVE_MAP_H
