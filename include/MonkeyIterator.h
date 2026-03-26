//
// Created by komozoi on 22.2.25.
//

#ifndef EXCESSIVE_MONKEYITERATOR_H
#define EXCESSIVE_MONKEYITERATOR_H


template<class T, class E>
class MonkeyIterator {
public:
	MonkeyIterator(const MonkeyIterator& other)
			: underlyingIterator(other.underlyingIterator) {}
	MonkeyIterator(MonkeyIterator&& other) noexcept
			: underlyingIterator(other.underlyingIterator) {}
	MonkeyIterator(const T& underlyingIterator)
			: underlyingIterator(underlyingIterator) {
	}

	MonkeyIterator& operator++() {
		underlyingIterator++;
		return *this;
	}

	bool operator==(const MonkeyIterator& rhs) {
		return underlyingIterator == rhs.underlyingIterator;
	}

	bool operator!=(const MonkeyIterator& rhs) {
		return underlyingIterator != rhs.underlyingIterator;
	}

	E operator*() { return (E)*underlyingIterator; }

	~MonkeyIterator() = default;

private:
	T underlyingIterator;
};


#endif //EXCESSIVE_MONKEYITERATOR_H
