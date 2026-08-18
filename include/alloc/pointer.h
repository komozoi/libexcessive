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
#include <memory>
#include <type_traits>
#include <utility>

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
 *
 * The concrete control block embeds the managed object in the same allocation (see
 * `sp_pointer_details_concrete_t`) so there is a single heap allocation per managed instance.
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
 *
 * Object and control data share one allocation: `[sp_pointer_details_concrete_t = header + U data]`.
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

template<class T>
class wp;

template<class T, SpPointerType P>
class SpManaged;

/**
 * @brief Catch-all: types that do not inherit `SpManaged` need no ownership registration.
 *
 * When `T` inherits `SpManaged<Base, ...>`, ADL finds the friend overload declared in
 * `SpManaged`, which registers `weak_this_` even if the concrete type is a subclass of `Base`.
 */
template<class Y>
inline void sp_enable_weak_this(const void*, Y*, const sp<Y>&) {}

/**
 * @brief Trait to check if a type is a specialization of `sp`.
 */
template<typename T>
struct is_sp : std::false_type {};

template<typename T>
struct is_sp<sp<T>> : std::true_type {};

/**
 * @brief Whether `sp<From>` may alias-cast to `sp<To>` (shared control block).
 *
 * Allows:
 * - **Upcast / same type**: `From*` is implicitly convertible to `To*` (existing behavior).
 * - **Downcast**: `To` derives from polymorphic `From` (e.g. `sp<Super>` → `sp<Abstract>` when the
 *   dynamic type is a further concrete subclass). Validated at runtime with `dynamic_cast`.
 *
 * Const/volatile may be added but not removed.
 */
// is_polymorphic / is_base_of need complete types. Do not instantiate them
// when From and To are the same (self-referential `struct N { sp<N> next; }`).
template<typename From, typename To, bool Enable>
struct sp_can_downcast : std::false_type {};

template<typename From, typename To>
struct sp_can_downcast<From, To, true> {
	using FromBare = std::remove_cv_t<From>;
	using ToBare = std::remove_cv_t<To>;
	static constexpr bool value =
		std::is_base_of<FromBare, ToBare>::value &&
		std::is_polymorphic<FromBare>::value &&
		(std::is_const<From>::value ? std::is_const<To>::value : true) &&
		(std::is_volatile<From>::value ? std::is_volatile<To>::value : true);
};

template<typename From, typename To>
struct sp_can_cast {
private:
	using FromBare = std::remove_cv_t<From>;
	using ToBare = std::remove_cv_t<To>;

public:
	static constexpr bool upcast = std::is_convertible<From*, To*>::value;

	static constexpr bool downcast =
		!upcast &&
		sp_can_downcast<From, To, !std::is_same<FromBare, ToBare>::value>::value;

	static constexpr bool value = upcast || downcast;
};

template<typename From, typename To>
inline constexpr bool sp_can_cast_v = sp_can_cast<From, To>::value;


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
		try_accept_owner();
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
		try_accept_owner();
	}

	/**
	 * @brief Constructs a `UNIQUE` pointer by copying the provided value into a new managed block.
	 * @param value The value to initialize the managed object with.
	 */
	explicit sp(const T& value) : type(UNIQUE) {
		details = new sp_pointer_details_concrete_t<T>(value);
		try_accept_owner();
	}

	/**
	 * @brief Move constructor. Transfers ownership without changing reference counts.
	 * @param other The source `sp` instance.
	 */
	sp(sp&& other) noexcept : details(nullptr) {
		swap(other);
	}

	/**
	 * @brief Templated move constructor. Converts `sp<U>` → `sp<T>` when castable (upcast or downcast).
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 *
	 * Downcasts are checked with `dynamic_cast`; on failure this is null and `other` is left unchanged.
	 */
	template<typename U, typename = typename std::enable_if<sp_can_cast<U, T>::value>::type>
	sp(sp<U>&& other) noexcept : details(nullptr), type(NULLPTR) {
		if (!other.details)
			return;

		if constexpr (!std::is_convertible<U*, T*>::value) {
			// Downcast: validate dynamic type before stealing the control block
			if (dynamic_cast<T*>(other.get()) == nullptr)
				return;
		}

		details = (sp_pointer_details_t*)other.details;
		type = other.type;
		other.details = nullptr;
		other.type = NULLPTR;
	}

	/**
	 * @brief Move assignment operator. Transfers ownership without changing reference counts.
	 * @param other The source `sp` instance.
	 * @return Reference to this `sp` instance.
	 */
	sp& operator=(sp&& other) noexcept {
		sp tmp(std::move(other));
		swap(tmp);
		return *this;
	}

	/**
	 * @brief Templated move assignment. Converts `sp<U>` → `sp<T>` when castable (upcast or downcast).
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 * @return Reference to this sp instance.
	 *
	 * Downcasts are checked with `dynamic_cast`; on failure both sides are left unchanged.
	 * The source is taken before this pointer is released, so assignment from a member of the owned object is safe.
	 */
	template<typename U, typename = typename std::enable_if<sp_can_cast<U, T>::value>::type>
	sp& operator=(sp<U>&& other) noexcept {
		sp tmp(std::move(other));
		if (!tmp && other)
			return *this;
		swap(tmp);
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
	 * @brief Templated copy constructor. Converts `sp<U>` → `sp<T>` when castable (upcast or downcast).
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 *
	 * Enables `sp<Abstract>(superSp)` and C-style `(sp<Abstract>)superSp` for polymorphic downcasts.
	 * Failed downcasts yield a null `sp`.
	 */
	template<typename U, typename = typename std::enable_if<sp_can_cast<U, T>::value>::type>
	sp(sp<U> const& other) {
		acquire_from_templated_copy(other);
	}

	/**
	 * @brief Copy assignment operator. Increments the reference count and releases current ownership.
	 *
	 * The source is copied before the old value is released, so assignment from a
	 * member of the owned object (`p = p.mut().next`) is safe.
	 * If the source pointer is `UNIQUE`, the new copy becomes `COPY_ON_WRITE`.
	 * @param other The source `sp` instance.
	 * @return Reference to this `sp` instance.
	 */
	sp& operator=(sp const& other) {
		sp tmp(other);
		swap(tmp);
		return *this;
	}

	/**
	 * @brief Templated copy assignment. Converts `sp<U>` → `sp<T>` when castable (upcast or downcast).
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 * @return Reference to this sp instance.
	 */
	template<typename U, typename = typename std::enable_if<sp_can_cast<U, T>::value>::type>
	sp& operator=(sp<U> const& other) {
		sp tmp(other);
		swap(tmp);
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

	/**
	 * @brief Compares this smart pointer to another smart pointer
	 *
	 * This compares the pointers themselves, as if they were raw pointers.
	 *
	 * @return `true` if both pointers are managing the same object, `false` otherwise.
	 */
	template <class U>
	bool operator==(const sp<U>& other) const {
		if (details == nullptr || other.details == nullptr)
			return details == (sp_pointer_details_t*)other.details;
		return details->getPtr() == other.details->getPtr();
	}

	/**
	 * @brief Compares this smart pointer to another smart pointer for inequality
	 *
	 * This compares the pointers themselves, as if they were raw pointers.
	 *
	 * @return `true` if both pointers are managing different objects, `false` otherwise.
	 */
	template <class U>
	bool operator!=(const sp<U>& other) const {
		return !(*this == other);
	}

	/**
	 * @brief Compares this smart pointer to a regular pointer
	 *
	 * This compares the pointers themselves, as if they were raw pointers.
	 *
	 * @return `true` if both pointers are managing the same object, `false` otherwise.
	 */
	template <class U>
	bool operator==(const U* other) const {
		return details ? details->getPtr() == (void*)other : other == nullptr;
	}

	/**
	 * @brief Compares this smart pointer to a regular pointer for inequality
	 *
	 * This compares the pointers themselves, as if they were raw pointers.
	 *
	 * @return `true` if both pointers are managing different objects, `false` otherwise.
	 */
	template <class U>
	bool operator!=(const U* other) const {
		return !(*this == other);
	}

	/**
	 * @brief Compares this smart pointer to nullptr
	 *
	 * @return `true` if this is a null pointer, `false` otherwise.
	 */
	bool operator==(nullptr_t) const {
		return details == nullptr;
	}

	/**
	 * @brief Compares this smart pointer to nullptr
	 *
	 * @return `true` if this pointer is valid, `false` otherwise.
	 */
	bool operator!=(nullptr_t) const {
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

	int numReferences() const {
		return details ? details->refs.load(std::memory_order_acquire) : 0;
	}

private:
	sp_pointer_details_t* details; /**< Pointer to the control block. */
	char type = NULLPTR;            /**< Current ownership model type. */

	/**
	 * @brief Internal constructor from raw details and type.
	 * @param d Pointer to the control block.
	 * @param t Initial ownership model type.
	 */
	sp(sp_pointer_details_t* d, SpPointerType t) : details(d), type(t) {
		try_accept_owner();
	}

	template<typename U>
	friend class sp;

	template<typename U>
	friend class wp;

	template<class U, SpPointerType P>
	friend class SpManaged;

	// ---------- Internals ----------

	/**
	 * @brief If `T` (or a base of `T`) inherits `SpManaged<Base, ...>`, register ownership so `ptr()` works.
	 *
	 * Uses ADL so a subclass managed as `sp<Derived>` still registers the `SpManaged<Base>` base.
	 */
	void try_accept_owner() {
		if (T* obj = get())
			sp_enable_weak_this(obj, obj, *this);
	}

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
	 * Handles reference counting, type transitions, and polymorphic downcast checks.
	 * @tparam U The type of the object managed by the source sp instance.
	 * @param other The source sp instance.
	 */
	template<typename U>
	void acquire_from_templated_copy(sp<U> const& other) {
		details = nullptr;
		type = NULLPTR;

		if (!other.details)
			return;

		// Upcast: implicit pointer conversion. Downcast: require dynamic type match.
		if constexpr (!std::is_convertible<U*, T*>::value) {
			if (dynamic_cast<T*>(other.get()) == nullptr)
				return;
		}

		details = (sp_pointer_details_t*)other.details;

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
		try_accept_owner();

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

/**
 * @brief Compares a regular pointer to a smart pointer
 *
 * This compares the pointers themselves, as if they were raw pointers.
 *
 * @return `true` if both pointers are managing the same object, `false` otherwise.
 */
template <typename T, typename U>
bool operator==(const T* lhs, const sp<U>& rhs) {
	return rhs == lhs;
}

/**
 * @brief Compares a regular pointer to a smart pointer for inequality
 *
 * This compares the pointers themselves, as if they were raw pointers.
 *
 * @return `true` if both pointers are managing different objects, `false` otherwise.
 */
template <typename T, typename U>
bool operator!=(const T* lhs, const sp<U>& rhs) {
	return !(rhs == lhs);
}

/**
 * @class wp
 * @tparam T The type of the object being observed.
 * @brief Non-owning weak observer of an `sp<T>`-managed object.
 *
 * `lock()` / `expired()` are only safe while at least one `sp` owns it
 */
template<class T>
class wp {
public:
	wp() : details(nullptr) {}

	wp(std::nullptr_t) : details(nullptr) {}

	/**
	 * @brief Constructs a weak observer from a strong pointer.
	 */
	wp(const sp<T>& owner) : details(owner.details) {}

	/**
	 * @brief Constructs a weak observer from a convertible strong pointer.
	 */
	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	wp(const sp<U>& owner) : details(owner.details) {}

	wp(const wp& other) = default;
	wp& operator=(const wp& other) = default;

	wp(wp&& other) noexcept : details(other.details) {
		other.details = nullptr;
	}

	wp& operator=(wp&& other) noexcept {
		details = other.details;
		other.details = nullptr;
		return *this;
	}

	wp& operator=(const sp<T>& owner) {
		details = owner.details;
		return *this;
	}

	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	wp& operator=(const sp<U>& owner) {
		details = owner.details;
		return *this;
	}

	wp& operator=(std::nullptr_t) {
		details = nullptr;
		return *this;
	}

	/**
	 * @brief Clears this weak observer.
	 */
	void reset() {
		details = nullptr;
	}

	/**
	 * @brief Returns true if there is no associated object or no remaining strong references.
	 *
	 * Only valid to call while the control block may still exist (e.g. from a live managed object).
	 */
	bool expired() const {
		return details == nullptr || details->refs.load(std::memory_order_acquire) == 0;
	}

	/**
	 * @brief Attempts to obtain a strong `sp<T>` if the object is still owned.
	 * @return A non-null `sp` if a strong reference could be taken; empty otherwise.
	 */
	sp<T> lock() const {
		if (!details)
			return sp<T>();

		uint32_t n = details->refs.load(std::memory_order_acquire);
		while (n != 0) {
			if (details->refs.compare_exchange_weak(n, n + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
				return sp<T>(details, SHARED);
			}
		}
		return sp<T>();
	}

	/**
	 * @brief True if this observer is associated with a control block (may still be expired).
	 */
	explicit operator bool() const {
		return details != nullptr;
	}

private:
	sp_pointer_details_t* details;

	template<typename U>
	friend class wp;

	template<typename U>
	friend class sp;
};

/**
 * @class SpManaged
 * @tparam T The type registered with `wp` / returned by `ptr()`
 * @tparam P Ownership model used when this type is first adopted by an `sp`
 * @brief Base class enabling an object to recover an `sp` to itself (like `enable_shared_from_this`).
 *
 * Inherit as: `class Foo : public SpManaged<Foo, SHARED> { ... };`
 * Subclasses of `Foo` inherit `ptr()` and receive `sp<Foo>` (the superclass type). Construct with
 * `sp<Derived>` / `sp<Foo>`; ownership registration works for the whole hierarchy via ADL.
 *
 * `weak_this_` is stored inside the object (same allocation as the control block + managed type).
 * Copy/move of the managed value does not transfer ownership registration; the smart-pointer
 * machinery re-registers when a new block is first owned.
 */
template<class T, SpPointerType P = SHARED>
class SpManaged {
private:
	mutable wp<T> weak_this_;

protected:
	SpManaged() = default;

	/** Copy does not share ownership registration; leave `weak_this_` empty. */
	SpManaged(const SpManaged&) {}

	SpManaged& operator=(const SpManaged&) {
		return *this;
	}

	/** Move does not transfer ownership registration; leave `weak_this_` empty. */
	SpManaged(SpManaged&&) noexcept {}

	SpManaged& operator=(SpManaged&&) noexcept {
		return *this;
	}

public:
	using __sp_managed_marker = T;
	static constexpr SpPointerType __sp_managed_pointer_type = P;

	/**
	 * @brief Returns a strong pointer to this object as `sp<T>` (the `SpManaged` CRTP type).
	 * @throws std::bad_weak_ptr if this object is not currently owned by an `sp`.
	 */
	sp<T> ptr() {
		sp<T> strong = weak_this_.lock();
		if (!strong)
			throw std::bad_weak_ptr();
		return strong;
	}

	/**
	 * @brief Const overload which prevents mutation of this object
	 * @throws std::bad_weak_ptr if this object is not currently owned by an `sp`.
	 */
	sp<T> ptr() const {
		sp<T> strong = weak_this_.lock();
		if (!strong)
			throw std::bad_weak_ptr();

		// Returns a non-const pointer, which copies on modify
		return strong.getWritableCopy();
	}

private:
	/**
	 * @brief Called when ownership is first taken. Accepts `sp` to `T` or any type convertible to `T*`.
	 */
	template<class U>
	void __accept_owner(const sp<U>& owner) const {
		static_assert(std::is_convertible<U*, T*>::value,
			"sp type must be convertible to SpManaged's registered type");
		if (weak_this_.expired())
			weak_this_ = owner;
	}

	/**
	 * @brief ADL-found registration for this `SpManaged` specialization (including subclasses of `T`).
	 *
	 * `T` is fixed by the `SpManaged` instantiation (not deduced from the argument), so
	 * `sp<Derived>` still registers when `Derived` inherits `T` / `SpManaged<T>`.
	 */
	template<class Y>
	friend void sp_enable_weak_this(const SpManaged* managed, Y* /*y*/, const sp<Y>& owner) {
		if (managed)
			managed->__accept_owner(owner);
	}

	template<class U>
	friend class sp;
};

#endif //LIBEXCESSIVE_POINTER_H
