//
// Created by nuclaer on 24.2.26.
//

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
