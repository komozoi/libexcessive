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


template<class T>
class Set {
public:
	/**
	 * Adds an item to the set
	 * @param item Item to add
	 * @return `true` if item is already present, `false` otherwise.
	 */
	virtual bool add(T item) = 0;
	virtual void addMany(const T* values, int count) {
		for (int i = 0; i < count; i++)
			add(values[i]);
	}

	virtual bool contains(T query) const = 0;

	virtual bool remove(T key) = 0;
	virtual void clear() = 0;

	virtual ~Set() = default;
};


#endif //LIBEXCESSIVE_SET_H
