//
// Created by nuclaer on 8/5/23.
//

#ifndef EXCESSIVE_SLABALLOCATOR_H
#define EXCESSIVE_SLABALLOCATOR_H


#include <cstdlib>
#include <cstdint>
#include <atomic>
#include <mutex>
#include "LinkedList.h"
#include "allocation/Allocator.h"


typedef struct slab_chunk_s {
	size_t blockWidth;
	int numBlocks;
	int lastFreeBlockIndex;
	char* blocks;
	// This is an array of bitmasks.
	uint32_t freeBlocks[];
} slab_chunk_t;


/**
 * This will be a better way of managing the memory of expressions and similar objects, for which there are many that
 * are frequently created, copied, and destroyed.
 */
class SlabAllocator : public Allocator {
public:
	SlabAllocator();

	void* alloc(size_t size) override;

	/**
	 * Works like normal free(): accepts nullptr, but will abort if an invalid pointer is provided
	 * @param ptr the memory to free
	 */
	void free(const void* ptr) override;

	/**
	 * Will check if the memory is allocated in this slab allocator.  If it is, then it works like normal free(), but
	 * returns true.  If the pointer is not in the slab allocator, then it returns false and has no effect.
	 *
	 * To be clear, if an attempt is made to free memory between blocks, or an attempt to free a free block, then
	 * abort() will be called.  Obviously the function cannot return in such a case.
	 *
	 * @param ptr the memory to free
	 * @return true if the memory is in the slab allocator and represents a valid block, false if not in slab allocator.
	 */
	bool freeIfPresent(const void* ptr);

	void prepare(int count, int size);

	template <class T>
	void prepare(int count) {
		prepare(count, sizeof(T));
	}

	/**
	 * Checks for and frees any chunks for which no blocks are allocated
	 */
	void destroyUnallocatedChunks();

	/**
	 * @return The number of bytes this object has reserved to be allocated.  This includes bytes already allocated.
	 */
	size_t getBytesControlled();

	/**
	 * @return The number of bytes allocated by alloc() and functions that use alloc() (includes clone, allocate, etc)
	 */
	size_t getBytesAllocated();

	/**
	 * @return The number of blocks this SlabAllocator controls.  This includes both allocated and free blocks.
	 */
	uint32_t countBlocks();

	/**
	 * @return The number of chunks this SlabAllocator controls.
	 */
	uint32_t countChunks();

	~SlabAllocator();

	bool contains(const void* ptr);

	void freeAll();

private:
	slab_chunk_t* getChunkOfMemory(const void* ptr);
	slab_chunk_t* getNextChunkForSize(size_t size);

	// Consider changing this structure later to be more efficient
	LinkedList<slab_chunk_t*> chunks;
	std::mutex lock;

	void internalFree(slab_chunk_t* chunk, const void* ptr);
};


// Global slab allocator for other parts of the program to use
extern SlabAllocator slab;


// Slab smart pointer (more efficient for large numbers of objects of the same type)
template<typename T>
class SlabPointer {
private:
	class RefCountNodeBase {
	public:
		std::atomic<int> ref_count;

		RefCountNodeBase() : ref_count(1) {}
		RefCountNodeBase(RefCountNodeBase&& other)  noexcept : ref_count((int)other.ref_count) {}

		virtual void destroy() = 0;
		virtual T* get() = 0;
	};

	class RefCountNodePtr : public RefCountNodeBase {
	public:
		T* ptr;

		explicit RefCountNodePtr(T* ptr) : ptr(ptr) {}

		void destroy() override {
			if (ptr) {
				ptr->~T();
				slab.free(ptr);
			}
			slab.free(this);
		}

		T* get() override {
			return ptr;
		}
	};

	template<typename T2>
	class RefCountNodeIns : public RefCountNodeBase {
	public:
		T2 val;

		explicit RefCountNodeIns(T2&& val) : val(val) {}
		explicit RefCountNodeIns(const T2& val) : val(val) {}

		void destroy() override {
			val.~T2();
			slab.free(this);
		}

		T* get() override {
			return &val;
		}
	};

	RefCountNodeBase* ref_node;

public:
	// Constructor
	SlabPointer() : ref_node(nullptr) {}

	explicit SlabPointer(T* ptr) {
		ref_node = slab.allocate<RefCountNodePtr>();
		new (ref_node) RefCountNodePtr(ptr);
	}

	template<typename T2, typename std::enable_if<std::is_base_of<T, T2>::value>::type* = nullptr>
	SlabPointer(T2&& val) {
		ref_node = slab.allocate<RefCountNodeIns<T2>>();
		new (ref_node) RefCountNodeIns<T2>(std::move(val));
	}

	template<typename T2, typename std::enable_if<std::is_base_of<T, T2>::value>::type* = nullptr>
	SlabPointer(const T2& val) {
		ref_node = slab.allocate<RefCountNodeIns<T2>>();
		new (ref_node) RefCountNodeIns<T2>(val);
	}

	SlabPointer(T&& val) {
		ref_node = slab.allocate<RefCountNodeIns<T>>();
		new (ref_node) RefCountNodeIns<T>(std::move(val));
	}

	SlabPointer(const T& val) {
		ref_node = slab.allocate<RefCountNodeIns<T>>();
		new (ref_node) RefCountNodeIns<T>(val);
	}

	// Copy Constructor
	SlabPointer(const SlabPointer& other) {
		if (other.ref_node == nullptr) {
			ref_node = nullptr;
			return;
		}

		ref_node = other.ref_node;
		ref_node->ref_count.fetch_add(1, std::memory_order_relaxed);
	}

	// Move Constructor
	SlabPointer(SlabPointer&& other) noexcept {
		ref_node = other.ref_node;
		other.ref_node = nullptr;
	}

	// Destructor
	~SlabPointer() {
		if (ref_node != nullptr) {
			int count = ref_node->ref_count.fetch_sub(1, std::memory_order_relaxed);
			if (count == 1)
				ref_node->destroy();
		}
	}

	// Copy Assignment
	SlabPointer& operator=(const SlabPointer& other) {
		if (this->ref_node == other.ref_node) return *this;

		if (other.ref_node != nullptr)
			other.ref_node->ref_count.fetch_add(1, std::memory_order_relaxed);

		if (ref_node != nullptr) {
			int count = ref_node->ref_count.fetch_sub(1, std::memory_order_relaxed);
			if (count == 1)
				ref_node->destroy();
		}

		ref_node = other.ref_node;

		return *this;
	}

	// Move Assignment
	SlabPointer& operator=(SlabPointer&& other) noexcept {
		if (this == &other) return *this;
		if (this->ref_node == other.ref_node) return *this;

		if (ref_node != nullptr) {
			int count = ref_node->ref_count.fetch_sub(1, std::memory_order_relaxed);
			if (count == 1)
				ref_node->destroy();
		}

		ref_node = other.ref_node;
		other.ref_node = nullptr;

		return *this;
	}

	// Reset
	void reset(T* ptr = nullptr) {
		if (ref_node != nullptr) {
			int count = ref_node->ref_count.fetch_sub(1, std::memory_order_relaxed);
			if (count == 1)
				ref_node->destroy();
		}

		if (ptr != nullptr) {
			ref_node = slab.allocate<RefCountNodePtr>();
			new (ref_node) RefCountNodePtr(ptr);
			ref_node->ref_count = 1;
		} else {
			ref_node = nullptr;
		}
	}

	int refCount() {
		return ref_node->ref_count.load();
	}

	// Dereference
	T& operator*() const {
		return *ref_node->get();
	}

	// Member Access
	T* operator->() const {
		return ref_node->get();
	}

	// Null Check
	explicit operator bool() const {
		return ref_node != nullptr && ref_node->get() != nullptr;
	}

	// Get Pointer
	T* get() const {
		return ref_node ? ref_node->get() : nullptr;
	}
};


#endif //EXCESSIVE_SLABALLOCATOR_H
