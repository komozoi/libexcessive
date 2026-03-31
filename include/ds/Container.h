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


template <typename T, typename R, typename Iterator, typename ConstIterator>
class Container {
public:
	virtual Iterator begin() = 0;
	virtual Iterator end() = 0;
	virtual ConstIterator begin() const = 0;
	virtual ConstIterator end() const = 0;
	ConstIterator cbegin() const { return begin(); }
	ConstIterator cend() const { return end(); }

	typedef std::reverse_iterator<Iterator> reverse_iterator;
	typedef std::reverse_iterator<ConstIterator> const_reverse_iterator;

	reverse_iterator rbegin() { return reverse_iterator(end()); }
	reverse_iterator rend() { return reverse_iterator(begin()); }
	const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
	const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
	const_reverse_iterator crbegin() const { return const_reverse_iterator(cend()); }
	const_reverse_iterator crend() const { return const_reverse_iterator(cbegin()); }

	virtual int size() const = 0;
	virtual bool isEmpty() const { return size() == 0; }

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