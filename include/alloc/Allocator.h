/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-01-02
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


#ifndef EXCESSIVE_ALLOCATOR_H
#define EXCESSIVE_ALLOCATOR_H

#include "string.h"
#include "stdlib.h"
#include <utility>


class Allocator {
public:

	template<class T>
	T* allocate(const T& value) {
		T* allocated = (T*)alloc(sizeof(T));
		new(allocated) T(value);
		return allocated;
	}

	template<class T>
	T* allocate(T&& value) {
		T* allocated = (T*)alloc(sizeof(T));
		new(allocated) T(std::move(value));
		return allocated;
	}

    /*template<class T>
    inline T* allocate(T obj) {
        T* out = (T*) alloc(sizeof(T));
        new(out) T(obj);
        return out;
    }*/

    template<class T>
    inline T* allocate() {
        return (T*) alloc(sizeof(T));
    }

    template<class T>
    T* allocateArray(int length) {
        return (T*)alloc(sizeof(T) * length);
    }

    /**
     * This function will copy the memory at `value` to a newly allocated block from this slab allocator, returning the
     * new block.  Note that this does not work on arrays, only single objects.
     * @tparam T The type of object to copy.  This is used to compute the size.
     * @param value The memory or object to be copied
     * @return A newly allocated clone of `value`
     */
    template<class T>
    T* clone(T* value) {
        T* newBlock = (T*)alloc(sizeof(T));
        memcpy(newBlock, value, sizeof(T));
        return newBlock;
    }

    /**
     * This function will copy the memory at `value` to a newly allocated block from this slab allocator, returning the
     * new block.  This function is designed to copy arrays of values.  It does not perform a deep copy.
     * @tparam T The type of object to copy.  This is used to compute the size.
     * @param value The memory or object to be copied
     * @return A newly allocated clone of `value`
     */
    template<class T>
    T* cloneArray(T* value, int length) {
        T* newBlock = (T*)alloc(sizeof(T) * length);
        memcpy(newBlock, value, sizeof(T) * length);
        return newBlock;
    }

    virtual void free(const void* p) {
        // Do nothing by default
    }

	virtual void* realloc(void* p, size_t newSize) {
		// Not supported by default
		return nullptr;
	}

    virtual void* alloc(size_t size) = 0;

    virtual ~Allocator() = default;
};


class DefaultAllocator: public Allocator {
public:

	void free(const void* p) override { ::free((void*)p); }

	void* alloc(size_t size) override { return malloc(size); }

	void* realloc(void* p, size_t newSize) override { return ::realloc(p, newSize); }

};

//inline DefaultAllocator defaultAllocator;
extern Allocator& defaultAllocator;


#endif //EXCESSIVE_ALLOCATOR_H
