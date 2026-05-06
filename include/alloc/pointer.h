/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-02-19
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


#ifndef LIBEXCESSIVE_POINTER_H
#define LIBEXCESSIVE_POINTER_H

#include <atomic>

#include "Allocator.h"


/**
 * @enum SpPointerType
 * @brief Defines the ownership and sharing behavior of an `sp<T>` pointer.
 *
 * This type determines how the pointer behaves when copied or mutated, enabling
 * standard shared ownership, unique ownership, and Copy-On-Write (COW) semantics.
 */
enum SpPointerType {
	UNIQUE,         /**< Sole owner of the object. Copying a UNIQUE pointer creates a COPY_ON_WRITE copy; the original remains UNIQUE. */
	SHARED,         /**< Standard shared ownership. Multiple pointers share the same object; mutations affect all. */
	COPY_ON_WRITE,  /**< Sharing is permitted, but `mut()` will trigger a deep copy if other references exist. */
	NULLPTR         /**< Represents an empty or null pointer. */
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

/**
 * @struct sp_pointer_details_t
 * @brief Internal control block for `sp<T>`, managing the reference count and object storage.
 */
struct sp_pointer_details_t {
	std::atomic<uint32_t> refs; /**< Atomic reference counter. */

	virtual ~sp_pointer_details_t() = default;

	/**
	 * @brief Returns a pointer to the managed object stored immediately after this control block.
	 * @return A void pointer to the start of the object's memory.
	 */
	virtual void* getPtr() = 0;

	/**
	 * @brief Performs a deep copy of the managed object into a new control block.
	 * @return A pointer to the new `sp_pointer_details_t`.
	 */
	virtual sp_pointer_details_t* copy() = 0;
};

/**
 * @struct sp_pointer_details_concrete_t
 * @brief Concrete implementation of `sp_pointer_details_t` for a specific type `U`.
 * @tparam U The actual type of the managed object.
 */
template<typename U>
struct sp_pointer_details_concrete_t : public sp_pointer_details_t {
	U data;

	template<typename... Args>
	sp_pointer_details_concrete_t(Args&&... args) : data(std::forward<Args>(args)...) {
		refs.store(1, std::memory_order_relaxed);
	}

	void* getPtr() override {
		return &data;
	}

	sp_pointer_details_t* copy() override {
		if constexpr (std::is_copy_constructible<U>::value) {
			return new sp_pointer_details_concrete_t<U>(data);
		} else {
			return nullptr;
		}
	}
};

// Originally based on https://codereview.stackexchange.com/a/163857
template<class T>
class sp;

/**
 * @brief Trait to check if a type is a specialization of `sp`.
 */
template<typename T>
struct is_sp : std::false_type {};

template<typename T>
struct is_sp<sp<T>> : std::true_type {};


/**
 * @class sp
 * @tparam T The type of the object being managed.
 * @brief A smart pointer implementation supporting reference-counted lifecycle management and Copy-On-Write (COW).
 *
 * `sp<T>` provides versatile ownership models, including unique, shared, and COW semantics.
 * It uses a single-allocation strategy for the control block and the object to minimize overhead.
 *
 * @note This implementation uses `std::atomic` for thread-safe reference counting.
 */
template<class T>
class sp {
public:
	/**
	 * @brief Constructs an empty `sp` instance (null pointer).
	 */
	sp() : details(nullptr) {}

	/**
	 * @brief Constructs an empty `sp` instance (null pointer).
	 * @param std::nullptr_t Explicit null pointer.
	 */
	sp(std::nullptr_t) : details(nullptr) {}

	/**
	 * @brief Constructs a new `sp<T>` with a specific pointer type and forwards arguments to `T`'s constructor.
	 * @tparam Args Variadic template for constructor arguments.
	 * @param type The initial ownership behavior (e.g., `UNIQUE`, `SHARED`).
	 * @param args Arguments to be forwarded to the constructor of `T`.
	 */
	template<typename... Args>
	explicit sp(SpPointerType type, Args&&... args) : type(type) {
		details = new sp_pointer_details_concrete_t<T>(std::forward<Args>(args)...);
	}

	/**
	 * @brief Static factory method to create a new `SHARED` managed object.
	 * @tparam Args Variadic template for constructor arguments.
	 * @param args Arguments for `T`'s constructor.
	 * @return A new `sp<T>` instance managing the created object.
	 */
	template<typename... Args>
	static sp<T> create(Args&&... args) {
		sp_pointer_details_t* block = new sp_pointer_details_concrete_t<T>(std::forward<Args>(args)...);

		return {block, SHARED};
	}

	/**
	 * @brief Constructs a `UNIQUE` pointer by moving the provided value into a new managed block.
	 * @tparam U Type of the value to be moved.
	 * @param value The value to initialize the managed object with.
	 */
	template<typename U, typename = typename std::enable_if<!is_sp<typename std::decay<U>::type>::value>::type>
	explicit sp(U&& value) : type(UNIQUE) {
		details = new sp_pointer_details_concrete_t<T>(std::forward<U>(value));
	}

	/**
	 * @brief Constructs a `UNIQUE` pointer by copying the provided value into a new managed block.
	 * @param value The value to initialize the managed object with.
	 */
	explicit sp(const T& value) : type(UNIQUE) {
		details = new sp_pointer_details_concrete_t<T>(value);
	}

	/**
	 * @brief Move constructor. Transfers ownership without changing reference counts.
	 * @param other The source `sp` instance.
	 */
	sp(sp&& other) noexcept : details(nullptr) {
		swap(other);
	}

	/**
	 * @brief Templated move constructor. Allows conversion from sp<U> to sp<T> if U* is convertible to T*.
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 */
	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	sp(sp<U>&& other) noexcept : details((sp_pointer_details_t*)other.details), type(other.type) {
		other.details = nullptr;
		other.type = NULLPTR;
	}

	/**
	 * @brief Move assignment operator. Transfers ownership without changing reference counts.
	 * @param other The source `sp` instance.
	 * @return Reference to this `sp` instance.
	 */
	sp& operator=(sp&& other) noexcept {
		swap(other);
		return *this;
	}

	/**
	 * @brief Templated move assignment operator. Allows conversion from sp<U> to sp<T> if U* is convertible to T*.
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 * @return Reference to this sp instance.
	 */
	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	sp& operator=(sp<U>&& other) noexcept {
		reset();
		details = (sp_pointer_details_t*)other.details;
		type = other.type;
		other.details = nullptr;
		other.type = NULLPTR;
		return *this;
	}

	/**
	 * @brief Copy constructor. Increments the reference count.
	 *
	 * If the source pointer is `UNIQUE`, the new copy becomes `COPY_ON_WRITE`.
	 * @param other The source `sp` instance.
	 */
	sp(sp const& other) {
		acquire_from_copy(other);
	}

	/**
	 * @brief Templated copy constructor. Allows conversion from sp<U> to sp<T> if U* is convertible to T*.
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 */
	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	sp(sp<U> const& other) {
		acquire_from_templated_copy(other);
	}

	/**
	 * @brief Copy assignment operator. Increments the reference count and releases current ownership.
	 *
	 * If the source pointer is `UNIQUE`, the new copy becomes `COPY_ON_WRITE`.
	 * @param other The source `sp` instance.
	 * @return Reference to this `sp` instance.
	 */
	sp& operator=(sp const& other) {
		if (this != &other) {
			reset();
			acquire_from_copy(other);
		}
		return *this;
	}

	/**
	 * @brief Templated copy assignment operator. Allows conversion from sp<U> to sp<T> if U* is convertible to T*.
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 * @return Reference to this sp instance.
	 */
	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	sp& operator=(sp<U> const& other) {
		reset();
		acquire_from_templated_copy(other);
		return *this;
	}

	/**
	 * @brief Destructor. Decrements the reference count and destroys the object/frees memory if needed.
	 */
	~sp() {
		release_ref();
	}

	// ---------- Access ----------

	/**
	 * @brief Provides read-only access to the managed object via pointer.
	 * @return A constant pointer to the managed object.
	 */
	const T* operator->() const {
		return get();
	}

	/**
	 * @brief Provides read-only access to the managed object via reference.
	 * @return A constant reference to the managed object.
	 */
	const T& operator*() const {
		return *get();
	}

	/**
	 * @brief Provides mutable access to the object.
	 *
	 * If the pointer type is `COPY_ON_WRITE` and multiple references exist, this method
	 * performs a **deep copy** of the object before returning the reference.
	 * @return A mutable reference to the managed object.
	 */
	T& mut() {
		detach_if_needed();
		return *get();
	}

	/**
	 * @brief Returns the raw pointer to the managed object.
	 * @return Raw pointer or `nullptr` if empty.
	 */
	T* get() const {
		return details ? (T*)details->getPtr() : nullptr;
	}

	/**
	 * @brief Checks if the pointer is not null.
	 * @return `true` if managing an object, `false` otherwise.
	 */
	explicit operator bool() const {
		return details != nullptr;
	}

	// ---------- Modifiers ----------

	/**
	 * @brief Releases the current reference and sets the pointer to null.
	 */
	void reset() {
		release_ref();
		type = NULLPTR;
	}

	/**
	 * @brief Swaps the content and type of two `sp` instances.
	 * @param other The other `sp` instance to swap with.
	 */
	void swap(sp& other) noexcept {
		std::swap(details, other.details);
		std::swap(type, other.type);
	}

	/**
	 * @brief Creates a new reference explicitly marked as `COPY_ON_WRITE`.
	 * @return A new `sp<T>` instance in COW mode.
	 */
	sp<T> getWritableCopy() const {
		details->refs.fetch_add(1, std::memory_order_relaxed);
		return sp<T>(details, COPY_ON_WRITE);
	}

	/**
	 * @brief Immediately copies the underlying data, giving a new pointer of the given type.
	 * @return A new `sp<T>` instance in the specified `SpPointerType`.
	 */
	sp<T> copy(SpPointerType newType) const {
		return sp<T>(details->copy(), newType);
	}

	/**
	 * @brief Returns the current ownership model type.
	 * @return The current `SpPointerType`.
	 */
	SpPointerType pointerType() const {
		return (SpPointerType)type;
	}

private:
	sp_pointer_details_t* details; /**< Pointer to the control block. */
	char type = NULLPTR;            /**< Current ownership model type. */

	/**
	 * @brief Internal constructor from raw details and type.
	 * @param d Pointer to the control block.
	 * @param t Initial ownership model type.
	 */
	sp(sp_pointer_details_t* d, SpPointerType t) : details(d), type(t) {}

	template<typename U>
	friend class sp;

	// ---------- Internals ----------

	/**
	 * @brief Shared logic for copy construction and assignment.
	 * Handles reference counting and type transitions (e.g., UNIQUE to COW).
	 * @param other The source `sp` instance.
	 */
	void acquire_from_copy(sp const& other) {
		details = other.details;
		if (!details)
			return;

		if (other.type == UNIQUE) {
			// UNIQUE copied -> copy becomes COW, original stays UNIQUE
			type = COPY_ON_WRITE;
		} else
			type = other.type;

		details->refs.fetch_add(1, std::memory_order_relaxed);
	}

	/**
	 * @brief Shared logic for templated copy construction and assignment.
	 * Handles reference counting and type transitions.
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 */
	template<typename U>
	void acquire_from_templated_copy(sp<U> const& other) {
		details = (sp_pointer_details_t*)other.details;
		if (!details)
			return;

		if (other.type == UNIQUE) {
			// UNIQUE copied -> copy becomes COW, original stays UNIQUE
			type = COPY_ON_WRITE;
		} else
			type = other.type;

		details->refs.fetch_add(1, std::memory_order_relaxed);
	}

	/**
	 * @brief Internal method to decrement reference count and potentially destroy the object.
	 */
	void release_ref() {
		if (!details)
			return;

		if (details->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			delete details;
		}

		details = nullptr;
	}

	/**
	 * @brief Internal COW logic. Performs a deep copy if the pointer is COW and shared.
	 */
	void detach_if_needed() {
		if (!details)
			return;

		if (type != COPY_ON_WRITE)
			return;

		if (details->refs.load(std::memory_order_acquire) == 1)
			return;

		// Need deep copy
		sp_pointer_details_t* old = details;

		details = old->copy();
		type = SHARED;

		// drop old ref
		if (old->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			delete old;
		}
	}
};

/**
 * @brief Swaps the content of two `sp` instances.
 * @tparam T The type of the managed object.
 * @param lhs First `sp` instance.
 * @param rhs Second `sp` instance.
 */
template<typename T>
void swap(sp<T>& lhs, sp<T>& rhs) {
	lhs.swap(rhs);
}

#endif //LIBEXCESSIVE_POINTER_H
