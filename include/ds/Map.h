//
// Created by komozoi on 24.2.26.
//

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
