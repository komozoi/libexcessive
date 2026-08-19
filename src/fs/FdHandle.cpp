/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-02-22
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


#include "fs/FdHandle.h"

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <sys/poll.h>
#include "fcntl.h"
#include "unistd.h"

#include "ds/HashMap.h"
#include "ds/HashSet.h"
#include "ds/Queue.h"


//#define DEBUG

#ifdef DEBUG
#include "ArraySet.h"
#endif



class FdHandleState {
public:
	DefaultAllocator allocator;
	HashMap<int, FdHandleData*>* fileDescriptors = nullptr;
	std::mutex mutex;
	std::condition_variable cv;
	HashSet<int> closing = HashSet<int>(16);

	Map<int, FdHandleData*>* getFds() {
		if (fileDescriptors == nullptr)
			fileDescriptors = new HashMap<int, FdHandleData*>(64, allocator);
		return fileDescriptors;
	}

	~FdHandleState();
};

static FdHandleState state;


typedef struct {
	off_t where;
	uint32_t size;
	char content[];
} queued_write_t;


class FdHandleData {
public:
	FdHandleData(int fd, bool ownsFd = true) : fd(fd), ownsFd(ownsFd), refs(0) {}

	void incRefUnlocked() {
		refs++;
	}

	void incRef() {
		std::lock_guard<std::mutex> lock(state.mutex);
		refs++;
	}

	void decRef() {
		int key = fd;
		{
			std::unique_lock<std::mutex> lock(state.mutex);
			refs--;
			if (refs > 0 || key < 0)
				return;
			state.getFds()->remove(key);
			state.closing.add(key);
		}
		destroy();
		{
			std::lock_guard<std::mutex> lock(state.mutex);
			state.closing.remove(key);
			state.cv.notify_all();
		}
	}

	virtual ssize_t write(const void* value, size_t size) const {
		char* buffer = (char*)value;
		int totalWritten = 0;
		int remaining = (int)size;

		while (totalWritten < (int)size) {
			int res = (int)::write(fd, &buffer[totalWritten], remaining);

			if (res < 0)
				return res;
			else if (res == 0)
				break;

			totalWritten += res;
			remaining -= res;
		}

		return totalWritten;
	}

	void queueWrite(const void* value, size_t size, off_t where) {
		std::lock_guard<std::recursive_mutex> _(mutex);

		queued_write_t* last = writeQueue.empty() ? nullptr : writeQueue.peekLast();

		if (last && last->size + last->where == where) {
			// Merge them
			queued_write_t* newWrite = (queued_write_t*)realloc(last, size + last->size + sizeof(queued_write_t));
			if (newWrite) {
				memcpy(&newWrite->content[newWrite->size], value, size);
				newWrite->size += size;

				writeQueue.setLast(newWrite);
				return;
			}
		} else if (last && ((off_t)(where + size) == last->where)) {
			// Merge them
			queued_write_t* newWrite = (queued_write_t*)realloc(last, size + last->size + sizeof(queued_write_t));
			if (newWrite) {
				memmove(&newWrite->content[size], newWrite->content, newWrite->size);
				memcpy(newWrite->content, value, size);
				newWrite->size += size;
				newWrite->where = where;

				writeQueue.setLast(newWrite);
				return;
			}
		}

		queued_write_t* item = (queued_write_t*) malloc(sizeof(queued_write_t) + size);
		item->size = size;
		item->where = where;
		memcpy(item->content, value, size);

		writeQueue.add(item);
	}

	void flushWrites() {
		std::lock_guard<std::recursive_mutex> _(mutex);

		while (!writeQueue.empty()) {
			queued_write_t* op = writeQueue.pop();
			const char* buf = op->content;
			size_t done = 0;
			while (done < op->size) {
				ssize_t n = ::pwrite(fd, buf + done, op->size - done, op->where + (off_t)done);
				if (n <= 0)
					break;
				done += (size_t)n;
			}
			free(op);
		}
	}

	virtual ssize_t read(void* value, size_t size) {
		// Drain under the handle mutex, then drop it before read(2) so a
		// blocking socket/pipe read does not stall other users of this fd.
		flushWrites();

		char* buffer = (char*)value;
		int totalRead = 0;
		int remaining = (int)size;

		while (totalRead < (int)size) {
			int res = (int)::read(fd, &buffer[totalRead], remaining);

			if (res < 0) {
				if (totalRead > 0) return totalRead;
				return res;
			} else if (res == 0)
				break;

			totalRead += res;
			remaining -= res;
		}

		return totalRead;
	}

	bool waitForRead() {
		struct pollfd pfd {fd, POLLIN | POLLHUP | POLLERR | POLLNVAL, 0};

		while (true) {
			int res = poll(&pfd, 1, 100);

			if (getShouldClose())
				return false;

			if (res < 0)
				return false;
			else if (res == 0) {
				int flags = fcntl(fd, F_GETFL);
				if (flags != -1 && (flags & O_NONBLOCK))
					return false;
				continue;
			}

			if (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL))
				return true;
		}
	}

	/**
	 * @brief Writes all pending changes to disk.
	 * 
	 * This method should be overridden by subclasses to provide specific 
	 * synchronization logic (e.g., calling syncfs or fsync).
	 */
	virtual void sync() { }

	/**
	 * @brief Executes all queued operations and synchronizes changes to disk.
	 */
	void flush() {
		flushWrites();
		sync();
	}

	virtual bool getIsNew() const { return false; }
	virtual bool isFile() const { return false; }
	virtual bool isStream() const { return false; }
	virtual void markToClose() {}
	virtual bool getShouldClose() const { return false; }

	virtual void destroy() {
		if (slab.contains(this)) {
			this->~FdHandleData();
			slab.free(this);
		} else
			delete this;
	}

	virtual ~FdHandleData() {
		if (fd >= 0) {
			flush();
			if (ownsFd)
				::close(fd);
		}
		fd = -1;
	}

	int fd;
	bool ownsFd;
	std::recursive_mutex mutex;

	int numReferences() const {
		return refs;
	}

private:
	std::atomic<int> refs;

	Queue<queued_write_t*> writeQueue;
};


class FileHandleData: public FdHandleData {
public:
	FileHandleData(int fd, bool isNew) : FdHandleData(fd), isNew(isNew) {}

	ssize_t write(const void* value, size_t size) const override {
		ssize_t o = FdHandleData::write(value, size);
		#ifdef DEBUG
			off_t where = lseek(fd, 0, SEEK_CUR);
			getHandleData(fd).writtenPositions.add(where);
			printf("WR %#06lx (%lu of %lu) <- ", where, o, size);
			if ((int)o == -1)
				printf("FAILURE: %s\n", strerror(errno));
			else {
				for (size_t i = 0; i < o; i++) {
					if (i != 0 && (i & 31) == 0)
						printf("\n                        ");
					else if (i != 0 && (i & 7) == 0)
						printf(" ");
					printf(" %02hhx", ((const char*) value)[i]);
				}
				printf("\n");
			}
		#endif
		return o;
	}

	ssize_t read(void* value, size_t size) override {
		ssize_t o = FdHandleData::read(value, size);
		#ifdef DEBUG
			off_t where = lseek(fd, 0, SEEK_CUR);
			if (isNew() && !getHandleData(fd).writtenPositions.contains(where))
				printf("READING UNINITIALIZED PART OF FILE (%lu):\n", where);
			printf("RD %#06lx (%lu of %lu) -> ", where, o, size);
			if ((int)o == -1)
				printf("FAILURE: %s\n", strerror(errno));
			else {
				for (size_t i = 0; i < o; i++) {
					if (i != 0 && (i & 31) == 0)
						printf("\n                        ");
					else if (i != 0 && (i & 7) == 0)
						printf(" ");
					printf(" %02hhx", ((const char*) value)[i]);
				}
				printf("\n");
			}
		#endif
		return o;
	}

	bool getIsNew() const override { return isNew; }
	bool isFile() const override { return true; }

	void sync() override {
#if defined(__APPLE__)
		::fsync(fd);
#else
		::fdatasync(fd);
#endif
	}

	const bool isNew;

#ifdef DEBUG
	ArraySet<off_t> writtenPositions;
#endif
};


class SocketHandleData: public FdHandleData {
public:
	explicit SocketHandleData(int fd, bool ownsFd = true) : FdHandleData(fd, ownsFd), shouldClose(false) {}

	ssize_t read(void* value, size_t size) override {
		struct pollfd pfd {fd, POLLIN | POLLHUP | POLLERR | POLLNVAL, 0};

		char* buffer = (char*)value;
		size_t totalRead = 0;
		size_t remaining = size;

		while (totalRead < size) {
			int res = poll(&pfd, 1, 100);

			if (shouldClose)
				return (ssize_t)totalRead;

			if (res < 0) {
				if (totalRead > 0) return (ssize_t)totalRead;
				return res;
			} else if (res == 0) {
				if (totalRead > 0) break;

				// Check if non-blocking
				int flags = fcntl(fd, F_GETFL);
				if (flags != -1 && (flags & O_NONBLOCK))
					return 0; // Return 0 for non-blocking if no data available

				continue;
			}

			if (pfd.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) {
				res = (int)::read(fd, &buffer[totalRead], remaining);

				if (res < 0) {
					if (totalRead > 0) return (ssize_t)totalRead;
					return res;
				} else if (res == 0)
					break;

				totalRead += res;
				remaining -= res;

				// Return immediately after reading some data for stream handles
				break;
			}
		}

		return (ssize_t)totalRead;
	}

	bool isStream() const override { return true; }
	void markToClose() override { if (fd > 0) shouldClose = true; }
	bool getShouldClose() const override { return shouldClose; }

	bool shouldClose;
};

static FileHandleData errorHandle(-1, false);

/** Caller must hold state.mutex. */
static FdHandleData& getHandleDataLocked(int fd) {
	if (fd < 0)
		return errorHandle;
	FdHandleData** slot = state.getFds()->getPtr(fd);
	if (slot == nullptr || *slot == nullptr)
		return errorHandle;
	return **slot;
}

static FdHandleData& getHandleData(int fd) {
	std::lock_guard<std::mutex> lock(state.mutex);
	return getHandleDataLocked(fd);
}

static void waitFdNotClosing(std::unique_lock<std::mutex>& lock, int fd) {
	while (state.closing.contains(fd))
		state.cv.wait(lock);
}

FdHandle::FdHandle(int fd) : fd(fd) {
	if (this->fd < 0)
		return;
	std::lock_guard<std::mutex> lock(state.mutex);
	FdHandleData& data = getHandleDataLocked(this->fd);
	if (&data != &errorHandle)
		data.incRefUnlocked();
}

FdHandle::FdHandle(const FdHandle& other) : fd(other.fd) {
	if (fd < 0)
		return;
	std::lock_guard<std::mutex> lock(state.mutex);
	FdHandleData& data = getHandleDataLocked(fd);
	if (&data != &errorHandle)
		data.incRefUnlocked();
}

FdHandle::FdHandle(FdHandle&& other) noexcept : fd(other.fd) {
	other.fd = -1;
}

FdHandle &FdHandle::operator=(FdHandle&& other) noexcept {
	if (fd >= 0)
		getHandleData(fd).decRef();
	fd = other.fd;
	other.fd = -1;
	return *this;
}

FdHandle &FdHandle::operator=(const FdHandle& other) {
	if (this == &other)
		return *this;

	if (fd >= 0)
		getHandleData(fd).decRef();
	fd = other.fd;
	if (fd >= 0)
		getHandleData(fd).incRef();
	return *this;
}

static void internNewFile(int fd, bool isNew) {
	std::unique_lock<std::mutex> lock(state.mutex);
	waitFdNotClosing(lock, fd);
	if (!state.getFds()->hasKey(fd)) {
		FileHandleData* handleData = new FileHandleData(fd, isNew);
		state.getFds()->put(fd, handleData);
	}
}

FdHandle FdHandle::open(const char* path, int mode) {
	int fd = ::open(path, mode, 0664);
	if (fd != -1)
		internNewFile(fd, false);
	return FdHandle(fd);
}

FdHandle FdHandle::open(const char* path, int mode, int flag) {
	bool isNew = false;
	int fd;

	if (mode & O_CREAT) {
		fd = ::open(path, mode ^ O_CREAT, flag);
		if (fd < 0) {
			isNew = true;
			fd = ::open(path, mode, flag);
		}
	} else
		fd = ::open(path, mode, flag);

	if (fd != -1)
		internNewFile(fd, isNew);
	return FdHandle(fd);
}

static void internExistingFd(int fd, bool ownsFd) {
	std::unique_lock<std::mutex> lock(state.mutex);
	waitFdNotClosing(lock, fd);
	if (!state.getFds()->hasKey(fd)) {
		SocketHandleData* handleData = new SocketHandleData(fd, ownsFd);
		state.getFds()->put(fd, handleData);
	}
}

FdHandle FdHandle::from(int fd) {
	if (fd >= 0)
		internExistingFd(fd, true);
	return FdHandle(fd);
}

FdHandle FdHandle::wrap(int fd) {
	if (fd >= 0)
		internExistingFd(fd, false);
	return FdHandle(fd);
}

void FdHandle::syncFile() const {
	getHandleData(fd).flush();
}

void FdHandle::syncFilesystem() const {
	getHandleData(fd).flushWrites();
#if defined(__linux__)
	syncfs(fd);
#else
	::fsync(fd);
#endif
}

void FdHandle::pipe(FdHandle& reader, FdHandle& writer) {
	int pipefd[2];

	if (::pipe(pipefd) != -1) {
		reader = from(pipefd[0]);
		writer = from(pipefd[1]);
	}
}

ssize_t FdHandle::write(const void* value, size_t size) const {
	return getHandleData(fd).write(value, size);
}

void FdHandle::queueWrite(const void* value, size_t size, off_t where) const {
	getHandleData(fd).queueWrite(value, size, where);
}

ssize_t FdHandle::read(void* value, size_t size) const {
	return getHandleData(fd).read(value, size);
}

ssize_t FdHandle::pread(void* value, size_t size, off_t offset) const {
	FdHandleData& handle = getHandleData(fd);

	// Drain any queued positional writes so the read sees the latest data,
	// then release the handle mutex before issuing the syscall.  Multiple
	// readers can then execute concurrently inside the kernel.
	{
		std::lock_guard<std::recursive_mutex> _(handle.mutex);
		handle.flushWrites();
	}

	char* buffer = (char*)value;
	size_t totalRead = 0;
	while (totalRead < size) {
		ssize_t res = ::pread(handle.fd, &buffer[totalRead], size - totalRead, offset + (off_t)totalRead);
		if (res < 0) {
			if (totalRead > 0) return (ssize_t)totalRead;
			return res;
		} else if (res == 0)
			break;
		totalRead += (size_t)res;
	}
	return (ssize_t)totalRead;
}

ssize_t FdHandle::pwrite(const void* value, size_t size, off_t offset) const {
	FdHandleData& handle = getHandleData(fd);

	{
		std::lock_guard<std::recursive_mutex> _(handle.mutex);
		handle.flushWrites();
	}

	const char* buffer = (const char*)value;
	size_t totalWritten = 0;
	while (totalWritten < size) {
		ssize_t res = ::pwrite(handle.fd, &buffer[totalWritten], size - totalWritten, offset + (off_t)totalWritten);
		if (res < 0) {
			if (totalWritten > 0) return (ssize_t)totalWritten;
			return res;
		} else if (res == 0)
			break;
		totalWritten += (size_t)res;
	}
	return (ssize_t)totalWritten;
}

bool FdHandle::waitForRead() const {
	return getHandleData(fd).waitForRead();
}

ssize_t FdHandle::printf(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	ssize_t result = vprintf(fmt, args);
	va_end(args);
	return result;
}

ssize_t FdHandle::vprintf(const char* fmt, va_list args) {
	char buffer[1024];
	va_list args_copy;
	va_copy(args_copy, args);
	int size = vsnprintf(buffer, sizeof(buffer), fmt, args_copy);
	va_end(args_copy);

	if (size < 0) return -1;

	if ((size_t)size < sizeof(buffer)) {
		return write(buffer, size);
	} else {
		char* dynamicBuffer = new char[size + 1];
		va_list args2;
		va_copy(args2, args);
		vsnprintf(dynamicBuffer, size + 1, fmt, args2);
		va_end(args2);
		ssize_t written = write(dynamicBuffer, size);
		delete[] dynamicBuffer;
		return written;
	}
}

bool FdHandle::readLine(std::string& line) {
	line.clear();
	char c;
	bool readAny = false;
	while (read(&c, 1) == 1) {
		readAny = true;
		if (c == '\n') break;
		line += c;
	}
	if (!line.empty() && line.back() == '\r') line.pop_back();
	return readAny;
}

off_t FdHandle::seek(off_t where, int whence) const {
	return lseek(fd, where, whence);
}

off_t FdHandle::seekToEndWithPadding(uint8_t paddingBytes) const {
	paddingBytes--;
	off_t off = lseek(fd, 0, SEEK_END);
	if ((off & paddingBytes) == 0)
		return off;
	return lseek(fd, (off + paddingBytes) & (0x7FFFFFFFFFFFFFFFL - paddingBytes), SEEK_SET);
}

bool FdHandle::isNew() const {
	return getHandleData(fd).getIsNew();
}

void FdHandle::flush() const {
	getHandleData(fd).flush();
}

void FdHandle::close() {
	if (fd >= 0) {
		getHandleData(fd).markToClose();
		getHandleData(fd).decRef();
		fd = -1;
	}
}

std::lock_guard<std::recursive_mutex> FdHandle::getLock() const {
	return std::lock_guard<std::recursive_mutex>(getHandleData(fd).mutex);
}

void FdHandle::markToClose() const {
	getHandleData(fd).markToClose();
}

// Grow `fd` to `newSize` bytes, allocating backing store for the new tail
// when possible so later MAP_SHARED writes do not contend on sparse
// allocation.  Falls back to ftruncate if preallocation is unavailable.
static bool growFileTo(int fd, off_t currentSize, off_t newSize) {
	if (newSize <= currentSize)
		return true;

#if defined(__APPLE__)
	fstore_t store{};
	store.fst_flags = F_ALLOCATECONTIG;
	store.fst_posmode = F_PEOFPOSMODE;
	store.fst_offset = 0;
	store.fst_length = newSize - currentSize;
	if (fcntl(fd, F_PREALLOCATE, &store) == -1) {
		store.fst_flags = F_ALLOCATEALL;
		fcntl(fd, F_PREALLOCATE, &store);
	}
#elif defined(__linux__)
	if (posix_fallocate(fd, currentSize, newSize - currentSize) == 0)
		return true;
#endif

	return ftruncate(fd, newSize) == 0;
}

MmapHandle FdHandle::getMmapHandle(off_t offset, size_t size, int prot, int flags) const {
	FdHandleData& handle = getHandleData(fd);

	// Make sure the file is big enough, then
	// restore the previous cursor position
	{
		std::lock_guard<std::recursive_mutex> _(handle.mutex);

		// Get current file offset to restore later
		off_t previousOffset = lseek(fd, 0, SEEK_CUR);
		if (previousOffset == -1)
			return MmapHandle(handle, nullptr, nullptr);

		handle.flushWrites();

		// Get file size
		off_t currentSize = lseek(fd, 0, SEEK_END);
		if (currentSize == -1) {
			lseek(fd, previousOffset, SEEK_SET);
			return MmapHandle(handle, nullptr, nullptr);
		}

		// Expand the file if needed.  ftruncate alone creates a sparse hole;
		// first-touch page faults then serialize on the filesystem allocator
		// (especially APFS MAP_SHARED).  Preallocate the new range when the
		// OS supports it, then set the size.
		off_t needed = offset + (off_t)size;
		if (currentSize < needed) {
			if (!growFileTo(fd, currentSize, needed)) {
				lseek(fd, previousOffset, SEEK_SET);
				return MmapHandle(handle, nullptr, nullptr);
			}
		}

		// Restore previous offset
		if (lseek(fd, previousOffset, SEEK_SET) == -1)
			return MmapHandle(handle, nullptr, nullptr);
	}

	char* mmapRegion = (char*)::mmap(nullptr, size, prot, flags, fd, offset);
	if (mmapRegion == MAP_FAILED)
		mmapRegion = nullptr;
	return MmapHandle(handle, mmapRegion, &mmapRegion[size]);
}

bool FdHandle::shouldClose() const {
	return getHandleData(fd).getShouldClose();
}

int FdHandle::numReferences() const {
	return getHandleData(fd).numReferences();
}

FdHandle::~FdHandle() {
	if (fd >= 0)
		getHandleData(fd).decRef();
}

FdTransaction::FdTransaction(const FdHandle& handle) : guard(getHandleData(handle.getFd()).mutex), handleData(getHandleData(handle.getFd())) {
}

ssize_t FdTransaction::write(const void* value, size_t size) const {
	return handleData.write(value, size);
}

ssize_t FdTransaction::read(void* value, size_t size) const {
	return handleData.read(value, size);
}

off_t FdTransaction::seek(off_t where, int whence) const {
	return lseek(handleData.fd, where, whence);
}

off_t FdTransaction::seekToEndWithPadding(uint8_t paddingBytes) const {
	paddingBytes--;
	off_t off = lseek(handleData.fd, 0, SEEK_END);
	if ((off & paddingBytes) == 0)
		return off;
	return lseek(handleData.fd, (off + 15) & (0x7FFFFFFFFFFFFFFFL - paddingBytes), SEEK_SET);
}

bool FdTransaction::isFile() const {
	return handleData.isFile();
}

bool FdTransaction::isStream() const {
	return handleData.isStream();
}


MmapHandle::MmapHandle(FdHandleData &handleData, char *data, char *end)
	: handleData(&handleData), data(data), cursor(data), end(end) {
	if (data)
		handleData.incRef();
}

MmapHandle& MmapHandle::operator=(MmapHandle&& other) noexcept {
	if (this == &other)
		return *this;
	if (handleData)
		handleData->decRef();
	handleData = other.handleData;
	data = other.data;
	cursor = other.cursor;
	end = other.end;
	other.handleData = nullptr;
	other.data = nullptr;
	other.cursor = nullptr;
	other.end = nullptr;
	return *this;
}

ssize_t MmapHandle::write(const void* value, size_t size) {
	if (size > (size_t)(end - cursor))
		size = end - cursor;
	memcpy(cursor, value, size);
	cursor = &cursor[size];
	return size;
}

ssize_t MmapHandle::read(void* value, size_t size) {
	if (size > (size_t)(end - cursor))
		size = end - cursor;
	memcpy(value, cursor, size);
	cursor = &cursor[size];
	return size;
}

off_t MmapHandle::seek(off_t where, int whence) {
	off_t pos = cursor - data;

	if (whence == SEEK_SET) {
		pos = where;
	} else if (whence == SEEK_CUR) {
		pos += where;
	} else if (whence == SEEK_END) {
		pos = (end - data) + where;
	} else
		return -1;

	if (pos < 0) pos = 0;
	if (pos > (end - data)) pos = end - data;

	cursor = &data[pos];

	return pos;
}

static void adviseRange(void* p, size_t bytes, int linuxAdvice, int posixAdvice) {
	if (!p || bytes == 0)
		return;
	uintptr_t addr = (uintptr_t)p;
	long page = sysconf(_SC_PAGESIZE);
	if (page <= 0)
		page = 4096;
	uintptr_t start = addr & ~((uintptr_t)page - 1);
	size_t len = (addr - start) + bytes;
	len = (len + (size_t)page - 1) & ~((size_t)page - 1);
#if defined(__linux__)
	(void)posixAdvice;
	madvise((void*)start, len, linuxAdvice);
#elif defined(POSIX_MADV_WILLNEED)
	(void)linuxAdvice;
	posix_madvise((void*)start, len, posixAdvice);
#else
	(void)linuxAdvice;
	(void)posixAdvice;
	(void)start;
	(void)len;
#endif
}

void memoryAdviseWillNeed(void* p, size_t bytes) {
#if defined(__linux__)
	adviseRange(p, bytes, MADV_WILLNEED, 0);
#elif defined(POSIX_MADV_WILLNEED)
	adviseRange(p, bytes, 0, POSIX_MADV_WILLNEED);
#else
	(void)p;
	(void)bytes;
#endif
}

void memoryAdviseSequential(void* p, size_t bytes) {
#if defined(__linux__)
	adviseRange(p, bytes, MADV_SEQUENTIAL, 0);
#elif defined(POSIX_MADV_SEQUENTIAL)
	adviseRange(p, bytes, 0, POSIX_MADV_SEQUENTIAL);
#else
	(void)p;
	(void)bytes;
#endif
}

void memoryAdviseHugePage(void* p, size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
	adviseRange(p, bytes, MADV_HUGEPAGE, 0);
#else
	(void)p;
	(void)bytes;
#endif
}

void MmapHandle::adviseWillNeed() const {
	if (data && end > data)
		memoryAdviseWillNeed(data, (size_t)(end - data));
}

void MmapHandle::adviseSequential() const {
	if (data && end > data)
		memoryAdviseSequential(data, (size_t)(end - data));
}

void MmapHandle::adviseHugePage() const {
	if (data && end > data)
		memoryAdviseHugePage(data, (size_t)(end - data));
}

MmapHandle::~MmapHandle() {
	if (data) {
		munmap(data, end - data);
		handleData->decRef();
	}
}

FdHandleState::~FdHandleState() {
	std::lock_guard<std::mutex> lock(mutex);
	if (fileDescriptors) {
		for (MapElement<int, FdHandleData*> entry: *fileDescriptors)
			entry.value->destroy();
		delete fileDescriptors;
	}
}
