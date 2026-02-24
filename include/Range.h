//
// Created by nuclaer on 19.2.25.
//

#ifndef EXCESSIVE_RANGE_H
#define EXCESSIVE_RANGE_H

#include <utility>


template <class T>
class Range {
public:
	Range(T&& begin, T&& end) : a(std::move(begin)), b(std::move(end)) {}
	Range(const T& begin, const T& end) : a(begin), b(end) {}

	T begin() { return a; }
	T end() { return b; }

	const T& begin() const { return a; }
	const T& end() const { return b; }

private:
	T a;
	T b;
};


#endif //EXCESSIVE_RANGE_H
