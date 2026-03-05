//
// Created by nuclaer on 19.2.26.
//

#ifndef LIBEXCESSIVE_POINTER_H
#define LIBEXCESSIVE_POINTER_H

#include "Allocator.h"


enum SpPointerType {
	UNIQUE,
	SHARED,
	COPY_ON_WRITE,
	NULLPTR
};

/*
struct compressed_pointer_details {
	int initialize(Allocator& allocator, size_t size);
	void* decode();

	void inc();
	void dec();

	uint32_t ptrIdx;
	uint16_t allocatorIdx;

};
*/

struct sp_pointer_details_t {
	std::atomic<uint32_t> refs;
	SpPointerType type;

	void* getPtr() {
		return (void*)&this[1];
	}
};

// Originally based on https://codereview.stackexchange.com/a/163857
template<class T>
class sp {
public:
	sp() : data(nullptr) {}

	explicit sp(T* data) : data(data) {}

	sp(std::nullptr_t) : data(nullptr) {}

	sp& operator=(std::nullptr_t) {
		reset();
		return* this;
	}

	sp(sp&& moving) noexcept: data(nullptr) {
		moving.swap(*this);
		// TODO: In the comments it was pointed out that this
		// does not match the implementation of std::unique_ptr
		// I am going to leave mine the same. But
		// the the standard provides some extra guarantees
		// and probably a more intuitive usage.
	}

	sp& operator=(sp&& moving) noexcept {
		moving.swap(*this);
		return* this;
	}

	template<typename U>
	sp(sp<U>&& moving) {
		sp<T> tmp(moving.release());
		tmp.swap(*this);
	}

	template<typename U>
	sp& operator=(sp<U>&& moving) {
		sp<T> tmp(moving.release());
		tmp.swap(*this);
		return* this;
	}

	sp(sp const& ) = delete;
	sp& operator=(sp const& ) = delete;

	T* operator->() const { return data; }
	T& operator*() const { return* data; }

	T* get() const { return data; }

	explicit operator bool() const { return data; }

	T* release() noexcept {
		T* result = nullptr;
		std::swap(result, data);
		return result;
	}

	void swap(sp& src) noexcept {
		std::swap(data, src.data);
	}

	void reset() {
		T* tmp = release();
		delete tmp;
	}

	~sp() {
		delete data;
	}

private:
	sp_pointer_details_t* details;
};

template<typename T>
void swap(sp<T>& lhs, sp<T>& rhs) {
	lhs.swap(rhs);
}

#endif //LIBEXCESSIVE_POINTER_H
