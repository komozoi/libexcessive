/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-03-26
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


#ifndef EXCESSIVELIB_FDHANDLE_H
#define EXCESSIVELIB_FDHANDLE_H

#include "unistd.h"
#include <mutex>
#include "sys/mman.h"


class FdHandleData;


/**
 * @brief Handle for memory-mapped file access.
 */
class MmapHandle {
public:
	/**
	 * @brief Constructs an MmapHandle.
	 * @param handleData Reference to the underlying FdHandleData.
	 * @param data Start address of the mapping.
	 * @param end End address of the mapping.
	 */
	explicit MmapHandle(FdHandleData& handleData, char* data, char* end);

	MmapHandle(const MmapHandle& other) = delete;

	/**
	 * @brief Move constructor for MmapHandle.
	 * @param other The MmapHandle to move from.
	 */
	inline MmapHandle(MmapHandle&& other) noexcept
		: handleData(other.handleData), data(other.data), cursor(other.cursor), end(other.end) {
		other.data = nullptr;
	}

	MmapHandle& operator=(MmapHandle&& other) noexcept = delete;

	/**
	 * @brief Checks if the mapping is valid.
	 * @return true if valid.
	 */
	inline operator bool() const { // NOLINT(*-explicit-constructor)
		return data != nullptr;
	}

	/**
	 * @brief Writes a value of type T to the current cursor position.
	 * @tparam T Type of the value.
	 * @param value The value to write.
	 * @return Number of bytes written, or -1 on error.
	 */
	template<class T>
	inline ssize_t write(const T& value) {
		return write(&value, sizeof(T));
	}

	/**
	 * @brief Writes raw bytes to the current cursor position.
	 * @param value Pointer to the data.
	 * @param size Number of bytes to write.
	 * @return Number of bytes written.
	 */
	ssize_t write(const void* value, size_t size);


	/**
	 * @brief Reads a value of type T from the current cursor position.
	 * @tparam T Type of the value.
	 * @param value Reference to store the read value.
	 * @return Number of bytes read.
	 */
	template<class T>
	inline ssize_t read(T& value) {
		return read(&value, sizeof(T));
	}

	/**
	 * @brief Reads and returns a value of type T from the current cursor position.
	 * @tparam T Type of the value.
	 * @return The read value.
	 */
	template<class T>
	inline T read() {
		T out;
		read(&out, sizeof(T));
		return out;
	}

	/**
	 * @brief Reads raw bytes from the current cursor position.
	 * @param value Pointer to the buffer.
	 * @param size Number of bytes to read.
	 * @return Number of bytes read.
	 */
	ssize_t read(void* value, size_t size);

	/**
	 * @brief Returns a direct pointer to the mapped memory.
	 * @tparam T Type of the pointer.
	 * @param where Offset from the start of the mapping.
	 * @return Pointer of type T*.
	 */
	template<typename T>
	inline T* directPointer(off_t where = 0) {
		return (T*)&data[where];
	}

	/**
	 * @brief Changes the current cursor position.
	 * @param where Offset.
	 * @param whence Reference point for the offset.
	 * @return New cursor position relative to the start of the mapping.
	 */
	off_t seek(off_t where, int whence = SEEK_SET);

	/**
	 * @brief Destructor for MmapHandle.
	 */
	~MmapHandle();

private:
	FdHandleData& handleData; /**< Underlying file handle data. */
	char* data;               /**< Start of the mapping. */
	char* cursor;             /**< Current position in the mapping. */
	char* end;                /**< End of the mapping. */
};


/**
 * @brief Reference-counted handle for a file descriptor.
 */
class FdHandle {
private:
	/**
	 * @brief Constructs an FdHandle from a raw file descriptor.
	 * @param fd Raw file descriptor.
	 */
	FdHandle(int fd); // NOLINT(google-explicit-constructor)

public:
	/**
	 * @brief Copy constructor (increases reference count).
	 * @param other Other FdHandle.
	 */
	FdHandle(const FdHandle& other);

	/**
	 * @brief Move constructor.
	 * @param other Other FdHandle.
	 */
	FdHandle(FdHandle&& other) noexcept;

	/**
	 * @brief Default constructor (invalid handle).
	 */
	inline FdHandle() { fd = -1; }

	/**
	 * @brief Move assignment operator.
	 * @param other Other FdHandle.
	 * @return Reference to this.
	 */
	FdHandle& operator=(FdHandle&& other) noexcept;

	/**
	 * @brief Checks if the handle is valid.
	 * @return true if valid.
	 */
	explicit inline operator bool () const noexcept {
		return fd != -1;
	}


	/**
	 * @brief Opens a file and returns an FdHandle.
	 * @param path Path to the file.
	 * @param mode Open mode (flags).
	 * @return FdHandle.
	 */
	static FdHandle open(const char* path, int mode);

	/**
	 * @brief Opens a file with specified flags and returns an FdHandle.
	 * @param path Path to the file.
	 * @param mode Open mode.
	 * @param flag Additional flags.
	 * @return FdHandle.
	 */
	static FdHandle open(const char* path, int mode, int flag);

	/**
	 * @brief Creates an FdHandle from an existing raw file descriptor.
	 * @param fd Raw file descriptor.
	 * @return FdHandle.
	 */
	static FdHandle from(int fd);

	/**
	 * @brief Creates a pipe and returns reader and writer FdHandles.
	 * @param reader Output reader handle.
	 * @param writer Output writer handle.
	 */
	static void pipe(FdHandle& reader, FdHandle& writer);

	/**
	 * @brief Writes a value of type T to the file.
	 * @tparam T Type of value.
	 * @param value Value to write.
	 * @return Number of bytes written.
	 */
	template<class T>
	inline ssize_t write(const T& value) const {
		return write(&value, sizeof(value));
	}

	/**
	 * @brief Writes raw bytes to the file.
	 * @param value Pointer to data.
	 * @param size Number of bytes to write.
	 * @return Number of bytes written.
	 */
	ssize_t write(const void* value, size_t size) const;

	/**
	 * @brief Queues a write operation at a specific offset.
	 * @tparam T Type of value.
	 * @param value Value to write.
	 * @param where Offset in the file.
	 */
	template<class T>
	void queueWrite(const T& value, off_t where) const {
		queueWrite(&value, sizeof(value), where);
	}

	/**
	 * @brief Queues a write operation with padding/alignment.
	 * @tparam T Type of value.
	 * @param value Value to write.
	 * @param where Offset in the file.
	 * @param alignment Required alignment.
	 */
	template<class T>
	void queueWriteWithPadding(const T& value, off_t where, uint16_t alignment) const {
		--alignment;
		queueWrite(&value, (sizeof(value) + alignment) & (0x7FFFFFFFFFFFFFFFL - alignment), where);
	}

	/**
	 * @brief Queues a raw write operation at a specific offset.
	 * @param value Pointer to data.
	 * @param size Number of bytes to write.
	 * @param where Offset in the file.
	 */
	void queueWrite(const void* value, size_t size, off_t where) const;

	/**
	 * @brief Reads a value of type T from the file.
	 * @tparam T Type of value.
	 * @param value Reference to store the read value.
	 * @return Number of bytes read.
	 */
	template<class T>
	inline ssize_t read(T& value) const {
		return read(&value, sizeof(value));
	}

	/**
	 * @brief Reads raw bytes from the file.
	 * @param value Pointer to buffer.
	 * @param size Number of bytes to read.
	 * @return Number of bytes read.
	 */
	ssize_t read(void* value, size_t size) const;

	/**
	 * @brief Waits until data is available for reading.
	 * @return true if ready.
	 */
	bool waitForRead() const;

	/**
	 * @brief Changes the file offset.
	 * @param where Offset.
	 * @param whence Reference point.
	 * @return New offset.
	 */
	off_t seek(off_t where, int whence = SEEK_SET) const;

	/**
	 * @brief Seeks to the end of the file with optional padding.
	 * @param paddingBytes Alignment padding.
	 * @return New offset.
	 */
	off_t seekToEndWithPadding(uint8_t paddingBytes) const;

	/**
	 * @brief Flushes any queued write operations.
	 */
	void flush() const;

	/**
	 * @brief Closes the file descriptor.
	 */
	void close();

	/**
	 * @brief Marks the handle to be closed.
	 */
	void markToClose() const;

	/**
	 * @brief Checks if the handle should be closed.
	 * @return true if it should close.
	 */
	bool shouldClose() const;

	/**
	 * @brief Checks if the file was newly created.
	 * @return true if new.
	 */
	bool isNew() const;

	/**
	 * @brief Returns the raw file descriptor.
	 * @return File descriptor.
	 */
	int getFd() const { return fd; }

	/**
	 * @brief Returns a lock guard for the file's mutex.
	 * @return lock_guard.
	 */
	std::lock_guard<std::mutex> getLock() const;

	/**
	 * @brief Destructor (decreases reference count).
	 */
	~FdHandle();

	/**
	 * @brief Creates a memory mapping for a portion of the file.
	 * @param offset Starting offset in the file.
	 * @param size Size of the mapping.
	 * @param prot Protection flags.
	 * @param flags Mapping flags.
	 * @return MmapHandle.
	 */
	MmapHandle getMmapHandle(off_t offset, size_t size, int prot = PROT_READ | PROT_WRITE, int flags = MAP_SHARED);

	/**
	 * @brief Returns the number of references to the underlying file.
	 * @return Reference count.
	 */
	int numReferences() const;

private:
	int16_t fd; /**< The raw file descriptor. */
};

/**
 * @brief Represents a transaction/locked access to an FdHandle.  An FdTransaction is atomic.
 */
class FdTransaction {
public:
	/**
	 * @brief Starts a transaction on an FdHandle.
	 * @param handle The FdHandle.
	 */
	explicit FdTransaction(const FdHandle& handle);

	FdTransaction(const FdTransaction& other) = delete;
	FdTransaction(FdTransaction&& other) noexcept = delete;

	FdTransaction& operator=(FdTransaction&& other) noexcept = delete;


	/**
	 * @brief Writes a value of type T during the transaction.
	 * @tparam T Type of value.
	 * @param value Value to write.
	 * @return Number of bytes written.
	 */
	template<class T>
	inline ssize_t write(const T& value) const {
		return write(&value, sizeof(T));
	}

	/**
	 * @brief Writes raw bytes during the transaction.
	 * @param value Pointer to data.
	 * @param size Number of bytes.
	 * @return Number of bytes written.
	 */
	ssize_t write(const void* value, size_t size) const;


	/**
	 * @brief Reads a value of type T during the transaction.
	 * @tparam T Type of value.
	 * @param value Reference to store the value.
	 * @return Number of bytes read.
	 */
	template<class T>
	inline ssize_t read(T& value) const {
		return read(&value, sizeof(T));
	}

	/**
	 * @brief Reads and returns a value of type T during the transaction.
	 * @tparam T Type of value.
	 * @return The value.
	 */
	template<class T>
	inline T read() const {
		T out;
		read(&out, sizeof(T));
		return out;
	}

	/**
	 * @brief Reads raw bytes during the transaction.
	 * @param value Pointer to buffer.
	 * @param size Number of bytes.
	 * @return Number of bytes read.
	 */
	ssize_t read(void* value, size_t size) const;


	/**
	 * @brief Changes the file offset during the transaction.
	 * @param where Offset.
	 * @param whence Reference point.
	 * @return New offset.
	 */
	off_t seek(off_t where, int whence = SEEK_SET) const;

	/**
	 * @brief Seeks to end with padding during the transaction.
	 * @param paddingBytes Padding.
	 * @return New offset.
	 */
	off_t seekToEndWithPadding(uint8_t paddingBytes) const;

	/**
	 * @brief Checks if the handle refers to a file.
	 * @return true if file.
	 */
	bool isFile() const;

	/**
	 * @brief Checks if the handle refers to a stream.
	 * @return true if stream.
	 */
	bool isStream() const;

	/**
	 * @brief Ends the transaction and releases the lock.
	 */
	~FdTransaction();

private:

	FdHandleData& handleData; /**< Underlying file handle data. */
};


#endif //EXCESSIVELIB_FDHANDLE_H
