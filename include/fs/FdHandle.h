//
// Created by komozoi on 26.3.25.
//

#ifndef EXCESSIVELIB_FDHANDLE_H
#define EXCESSIVELIB_FDHANDLE_H

#include "unistd.h"
#include <mutex>


class FdHandle {
private:
	FdHandle(int fd); // NOLINT(google-explicit-constructor)

public:
	FdHandle(const FdHandle& other);
	FdHandle(FdHandle&& other) noexcept;
	inline FdHandle() { fd = -1; }

	FdHandle& operator=(FdHandle&& other) noexcept;

	explicit inline operator bool () const noexcept {
		return fd != -1;
	}


	static FdHandle open(const char* path, int mode);
	static FdHandle open(const char* path, int mode, int flag);
	static FdHandle from(int fd);
	static void pipe(FdHandle& reader, FdHandle& writer);

	template<class T>
	inline ssize_t write(const T& value) const {
		return write(&value, sizeof(value));
	}

	ssize_t write(const void* value, size_t size) const;

	template<class T>
	void queueWrite(const T& value, off_t where) const {
		queueWrite(&value, sizeof(value), where);
	}

	template<class T>
	void queueWriteWithPadding(const T& value, off_t where, uint16_t alignment) const {
		--alignment;
		queueWrite(&value, (sizeof(value) + alignment) & (0x7FFFFFFFFFFFFFFFL - alignment), where);
	}

	void queueWrite(const void* value, size_t size, off_t where) const;

	template<class T>
	inline ssize_t read(T& value) const {
		return read(&value, sizeof(value));
	}

	ssize_t read(void* value, size_t size) const;

	bool waitForRead() const;

	off_t seek(off_t where, int whence = SEEK_SET) const;
	off_t seekToEndWithPadding(uint8_t paddingBytes) const;

	void flush() const;

	void close();
	void markToClose() const;
	bool shouldClose() const;

	bool isNew() const;
	int getFd() const { return fd; }

	std::lock_guard<std::mutex> getLock() const;

	~FdHandle();

private:
	int16_t fd;
};


class FdHandleData;

class FdTransaction {
public:
	explicit FdTransaction(const FdHandle& handle);

	FdTransaction(const FdTransaction& other) = delete;
	FdTransaction(FdTransaction&& other) noexcept = delete;

	FdTransaction& operator=(FdTransaction&& other) noexcept = delete;


	template<class T>
	inline ssize_t write(const T& value) const {
		return write(&value, sizeof(T));
	}

	ssize_t write(const void* value, size_t size) const;


	template<class T>
	inline ssize_t read(T& value) const {
		return read(&value, sizeof(T));
	}

	template<class T>
	inline T read() const {
		T out;
		read(&out, sizeof(T));
		return out;
	}

	ssize_t read(void* value, size_t size) const;


	off_t seek(off_t where, int whence = SEEK_SET) const;
	off_t seekToEndWithPadding(uint8_t paddingBytes) const;

	bool isFile() const;
	bool isStream() const;

	~FdTransaction();

private:

	FdHandleData& handleData;
};


#endif //EXCESSIVELIB_FDHANDLE_H
