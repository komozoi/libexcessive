//
// Created by komozoi on 19.2.26.
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

	void* getPtr() {
		return (void*)&this[1];
	}
};

// Originally based on https://codereview.stackexchange.com/a/163857
template<class T>
class sp {
public:
	sp() : details(nullptr) {}
	sp(std::nullptr_t) : details(nullptr) {}

	template<typename... Args>
	explicit sp(SpPointerType type, Args&&... args) : type(type) {
		size_t total = sizeof(sp_pointer_details_t) + sizeof(T);
		details = (sp_pointer_details_t*)::operator new(total);

		details->refs.store(1, std::memory_order_relaxed);

		new (details->getPtr()) T(std::forward<Args>(args)...);
	}

	template<typename... Args>
	static sp<T> create(Args&&... args) {
		size_t total = sizeof(sp_pointer_details_t) + sizeof(T);
		auto* block = (sp_pointer_details_t*)::operator new(total);

		block->refs.store(1, std::memory_order_relaxed);

		new (block->getPtr()) T(std::forward<Args>(args)...);

		return {block, SpPointerType::UNIQUE};
	}

	template<typename U>
	explicit sp(U&& value) : type(SpPointerType::UNIQUE) {
		size_t total = sizeof(sp_pointer_details_t) + sizeof(T);
		details = (sp_pointer_details_t*)::operator new(total);

		details->refs.store(1, std::memory_order_relaxed);

		new (details->getPtr()) T(std::forward<U>(value));
	}

	explicit sp(const T& value) : type(SpPointerType::UNIQUE) {
		size_t total = sizeof(sp_pointer_details_t) + sizeof(T);
		details = (sp_pointer_details_t*)::operator new(total);

		details->refs.store(1, std::memory_order_relaxed);

		new (details->getPtr()) T(value);
	}

	// Move
	sp(sp&& other) noexcept : details(nullptr) {
		swap(other);
	}

	sp& operator=(sp&& other) noexcept {
		swap(other);
		return *this;
	}

	// Copy (enables COW logic)
	sp(sp const& other) {
		acquire_from_copy(other);
	}

	sp& operator=(sp const& other) {
		if (this != &other) {
			reset();
			acquire_from_copy(other);
		}
		return *this;
	}

	~sp() {
		release_ref();
	}

	// ---------- Access ----------

	const T* operator->() const {
		return get();
	}

	const T& operator*() const {
		return *get();
	}

	T& mut() {
		detach_if_needed();
		return *get();
	}

	T* get() const {
		return details ? (T*)details->getPtr() : nullptr;
	}

	explicit operator bool() const {
		return details != nullptr;
	}

	// ---------- Modifiers ----------

	void reset() {
		release_ref();
		details = nullptr;
	}

	void swap(sp& other) noexcept {
		std::swap(details, other.details);
	}

	sp<T> getWritableCopy() const {
		return sp<T>(details, SpPointerType::COPY_ON_WRITE);
	}

	SpPointerType pointerType() const {
		return type;
	}

private:
	sp_pointer_details_t* details;
	SpPointerType type;

	sp(sp_pointer_details_t* details, SpPointerType type) : details(details), type(type) {}

	// ---------- Internals ----------

	void acquire_from_copy(sp const& other) {
		details = other.details;
		if (!details)
			return;

		if (other.type == SpPointerType::UNIQUE) {
			// UNIQUE copied -> becomes COW
			type = SpPointerType::COPY_ON_WRITE;
		} else
			type = other.type;

		details->refs.fetch_add(1, std::memory_order_relaxed);
	}

	void release_ref() {
		if (!details)
			return;

		if (details->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			// destroy object
			T* obj = (T*)details->getPtr();
			obj->~T();
			::operator delete(details);
		}

		details = nullptr;
	}

	void detach_if_needed() {
		if (!details)
			return;

		if (type != SpPointerType::COPY_ON_WRITE)
			return;

		if (details->refs.load(std::memory_order_acquire) == 1)
			return;

		// Need deep copy
		sp_pointer_details_t* old = details;

		size_t total = sizeof(sp_pointer_details_t) + sizeof(T);
		auto* block = (sp_pointer_details_t*)::operator new(total);

		block->refs.store(1, std::memory_order_relaxed);

		new (block->getPtr()) T(*(T*)old->getPtr());

		details = block;
		type = SpPointerType::SHARED;

		// drop old ref
		if (old->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			T* obj = (T*)old->getPtr();
			obj->~T();
			::operator delete(old);
		}
	}
};

template<typename T>
void swap(sp<T>& lhs, sp<T>& rhs) {
	lhs.swap(rhs);
}

#endif //LIBEXCESSIVE_POINTER_H
