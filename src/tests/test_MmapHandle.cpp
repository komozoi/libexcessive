
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <thread>
#include "fs/FdHandle.h"
#include "universaltime.h"


TEST(MmapHandleTest, BasicWrite) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 1024);

	const char test_str[] = "Hello, World!";
	size_t str_size = strlen(test_str);
	ssize_t written = mmap_handle.write(test_str, str_size);
	EXPECT_EQ(written, static_cast<ssize_t>(str_size));
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), static_cast<off_t>(str_size));

	msync(mmap_handle.directPointer<char>(0), 1024, MS_SYNC);

	FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
	MmapHandle read_mmap = read_handle.getMmapHandle(0, 1024, PROT_READ);
	char buffer[13];
	read_mmap.read(buffer, 13);
	EXPECT_EQ(memcmp(buffer, test_str, str_size), 0);

	unlink(temp_filename);
}

TEST(MmapHandleTest, TemplateWrite) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 1024);

	uint32_t value = 0xDEADBEEF;
	ssize_t written = mmap_handle.write(value);
	EXPECT_EQ(written, sizeof(value));

	uint32_t* ptr = mmap_handle.directPointer<uint32_t>();
	EXPECT_EQ(*ptr, value);

	unlink(temp_filename);
}

TEST(MmapHandleTest, WriteOverflow) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 5);

	const char test_str[] = "1234567890";
	ssize_t written = mmap_handle.write(test_str, strlen(test_str));
	EXPECT_EQ(written, 5);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 5);

	FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
	MmapHandle read_mmap = read_handle.getMmapHandle(0, 5, PROT_READ);
	char buffer[5];
	read_mmap.read(buffer, 5);
	EXPECT_EQ(memcmp(buffer, "12345", 5), 0);

	unlink(temp_filename);
}

TEST(MmapHandleTest, BasicRead) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle write_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle write_mmap = write_handle.getMmapHandle(0, 5);
	uint8_t init_data[5] = {1, 2, 3, 4, 5};
	write_mmap.write(init_data, 5);
	msync(write_mmap.directPointer<char>(0), 5, MS_SYNC);

	FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
	MmapHandle mmap_handle = read_handle.getMmapHandle(0, 5, PROT_READ);

	uint8_t buffer[5];
	ssize_t read_bytes = mmap_handle.read(buffer, 5);
	EXPECT_EQ(read_bytes, 5);
	EXPECT_EQ(memcmp(buffer, init_data, 5), 0);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 5);

	unlink(temp_filename);
}

TEST(MmapHandleTest, TemplateReadRef) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle write_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle write_mmap = write_handle.getMmapHandle(0, sizeof(uint32_t));
	uint32_t init_value = 0x12345678;
	write_mmap.write(init_value);
	msync(write_mmap.directPointer<char>(0), sizeof(uint32_t), MS_SYNC);

	FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
	MmapHandle mmap_handle = read_handle.getMmapHandle(0, sizeof(uint32_t), PROT_READ);

	uint32_t value;
	ssize_t read_bytes = mmap_handle.read(value);
	EXPECT_EQ(read_bytes, sizeof(value));
	EXPECT_EQ(value, init_value);

	unlink(temp_filename);
}

TEST(MmapHandleTest, TemplateReadReturn) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle write_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle write_mmap = write_handle.getMmapHandle(0, sizeof(uint32_t));
	uint32_t init_value = 0xABCDEF01;
	write_mmap.write(init_value);
	msync(write_mmap.directPointer<char>(0), sizeof(uint32_t), MS_SYNC);

	FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
	MmapHandle mmap_handle = read_handle.getMmapHandle(0, sizeof(uint32_t), PROT_READ);

	uint32_t value = mmap_handle.read<uint32_t>();
	EXPECT_EQ(value, init_value);

	unlink(temp_filename);
}

TEST(MmapHandleTest, ReadOverflow) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle write_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle write_mmap = write_handle.getMmapHandle(0, 3);
	uint8_t init_data[3] = {1, 2, 3};
	write_mmap.write(init_data, 3);
	msync(write_mmap.directPointer<char>(0), 3, MS_SYNC);

	FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
	MmapHandle mmap_handle = read_handle.getMmapHandle(0, 3, PROT_READ);

	uint8_t buffer[5] = {0};
	ssize_t read_bytes = mmap_handle.read(buffer, 5);
	EXPECT_EQ(read_bytes, 3);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 3);
	uint8_t expected[5] = {1, 2, 3, 0, 0};
	EXPECT_EQ(memcmp(buffer, expected, 5), 0);

	unlink(temp_filename);
}

TEST(MmapHandleTest, SeekSet) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 10);

	off_t new_pos = mmap_handle.seek(5, SEEK_SET);
	EXPECT_EQ(new_pos, 5);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 5);

	new_pos = mmap_handle.seek(15, SEEK_SET);
	EXPECT_EQ(new_pos, 10);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 10);

	new_pos = mmap_handle.seek(-3, SEEK_SET);
	EXPECT_EQ(new_pos, 0);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 0);

	unlink(temp_filename);
}

TEST(MmapHandleTest, SeekCur) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 10);

	mmap_handle.seek(3, SEEK_SET);
	off_t new_pos = mmap_handle.seek(4, SEEK_CUR);
	EXPECT_EQ(new_pos, 7);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 7);

	new_pos = mmap_handle.seek(5, SEEK_CUR);
	EXPECT_EQ(new_pos, 10);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 10);

	new_pos = mmap_handle.seek(-6, SEEK_CUR);
	EXPECT_EQ(new_pos, 4);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 4);

	new_pos = mmap_handle.seek(-10, SEEK_CUR);
	EXPECT_EQ(new_pos, 0);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 0);

	unlink(temp_filename);
}

TEST(MmapHandleTest, SeekEnd) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 10);

	off_t new_pos = mmap_handle.seek(0, SEEK_END);
	EXPECT_EQ(new_pos, 10);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 10);

	new_pos = mmap_handle.seek(-3, SEEK_END);
	EXPECT_EQ(new_pos, 7);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 7);

	new_pos = mmap_handle.seek(5, SEEK_END);
	EXPECT_EQ(new_pos, 10);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 10);

	new_pos = mmap_handle.seek(-15, SEEK_END);
	EXPECT_EQ(new_pos, 0);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), 0);

	unlink(temp_filename);
}

TEST(MmapHandleTest, SeekInvalidWhence) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 1024);

	off_t original_pos = mmap_handle.seek(0, SEEK_CUR);
	off_t new_pos = mmap_handle.seek(0, 999);
	EXPECT_EQ(new_pos, -1);
	EXPECT_EQ(mmap_handle.seek(0, SEEK_CUR), original_pos);

	unlink(temp_filename);
}

TEST(MmapHandleTest, DirectPointer) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);

	{
		FdHandle write_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
		MmapHandle write_mmap = write_handle.getMmapHandle(0, 4);
		uint8_t init_data[4] = {0xAA, 0xBB, 0xCC, 0xDD};
		write_mmap.write(init_data, 4);
		msync(write_mmap.directPointer<char>(0), 4, MS_SYNC);

		FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
		MmapHandle mmap_handle = read_handle.getMmapHandle(0, 4, PROT_READ);

		uint16_t *ptr = mmap_handle.directPointer<uint16_t>(1);
		EXPECT_EQ(*ptr, 0xCCBB);
	}

	{
		FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
		MmapHandle mmap_handle = read_handle.getMmapHandle(0, 4, PROT_READ);

		uint16_t *ptr = mmap_handle.directPointer<uint16_t>(1);
		EXPECT_EQ(*ptr, 0xCCBB);
	}

	unlink(temp_filename);
}

TEST(MmapHandleTest, CombinedOperations) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 20);

	const char str1[] = "First";
	mmap_handle.write(str1, 5);

	mmap_handle.seek(10, SEEK_SET);

	const char str2[] = "Second";
	mmap_handle.write(str2, 6);

	mmap_handle.seek(0, SEEK_SET);

	char buffer[5];
	mmap_handle.read(buffer, 5);
	EXPECT_EQ(memcmp(buffer, "First", 5), 0);

	mmap_handle.seek(10, SEEK_SET);

	char buffer2[6];
	mmap_handle.read(buffer2, 6);
	EXPECT_EQ(memcmp(buffer2, "Second", 6), 0);

	msync(mmap_handle.directPointer<char>(0), 20, MS_SYNC);

	FdHandle read_handle = FdHandle::open(temp_filename, O_RDONLY);
	MmapHandle read_mmap = read_handle.getMmapHandle(0, 20, PROT_READ);
	uint8_t contents[20];
	read_mmap.read(contents, 20);
	uint8_t expected[20] = {0};
	memcpy(expected, "First", 5);
	memcpy(expected + 10, "Second", 6);
	EXPECT_EQ(memcmp(contents, expected, 20), 0);

	unlink(temp_filename);
}

TEST(MmapHandleTest, ZeroSize) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap_handle = fd_handle.getMmapHandle(0, 0);
	EXPECT_FALSE(mmap_handle);

	unlink(temp_filename);
}

TEST(MmapHandleTest, WriteAndReadMmap) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);
	const char test_content[] = "Hello, World!";

	{
		FdHandle handle_write = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
		ASSERT_TRUE(handle_write);

		MmapHandle writer = handle_write.getMmapHandle(0, 1024);
		ASSERT_TRUE(writer);

		writer.write(test_content, strlen(test_content));
		writer.write<uint8_t>(0);
	}

	{
		FdHandle handle_read = FdHandle::open(test_file, O_RDONLY);
		MmapHandle reader = handle_read.getMmapHandle(0, 1024, PROT_READ, MAP_SHARED);

		EXPECT_STREQ(reader.directPointer<const char>(), test_content);
	}

	unlink(test_file);
}

TEST(MmapHandleTest, CloseOnReferenceLostMmap) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	ASSERT_TRUE(handle);
	EXPECT_EQ(handle.numReferences(), 1);
	{
		MmapHandle writer = handle.getMmapHandle(0, 1024);
		ASSERT_TRUE(writer);
		writer.write<int>(0);
		EXPECT_EQ(handle.numReferences(), 2);
	}
	EXPECT_EQ(handle.numReferences(), 1);

	unlink(test_file);
}

TEST(MmapHandleTest, TrackReferencesCorrectlyWhenHandleLost) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);

	// This allows extraction of the file descriptor and a MmapHandle,
	// while also destroying our original file handle object
	int extractedFd;
	MmapHandle* writer;
	{
		FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
		ASSERT_TRUE(handle);
		extractedFd = handle.getFd();
		writer = new MmapHandle(handle.getMmapHandle(0, 1024));
	}

	if (!*writer) {
		delete writer;
		ASSERT_TRUE(false);
	}

	// Expect that we can write to the handle and that the
	// file is still open
	writer->write<int>(0);
	EXPECT_NE(fcntl(extractedFd, F_GETFD), -1);

	// Expect that the file is closed when we delete the writer
	delete writer;
	EXPECT_EQ(fcntl(extractedFd, F_GETFD), -1);
	EXPECT_EQ(errno, EBADF);

	unlink(test_file);
}

TEST(MmapHandleTest, VerifyCloseAfterMmap) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);
	int extractedFd;
	{
		FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
		extractedFd = handle.getFd();
		{
			MmapHandle writer1 = handle.getMmapHandle(0, 1024);
			MmapHandle writer2 = handle.getMmapHandle(0, 1024);
			EXPECT_EQ(handle.numReferences(), 3);
		}
		EXPECT_EQ(handle.numReferences(), 1);
	}
	EXPECT_EQ(fcntl(extractedFd, F_GETFD), -1);
	EXPECT_EQ(errno, EBADF);

	unlink(test_file);
}

TEST(MmapHandleTest, ConcurrentMmapWrite) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	MmapHandle mmap1 = handle.getMmapHandle(0, 1024);
	MmapHandle mmap2 = handle.getMmapHandle(0, 1024);

	const char test_str[] = "Concurrent";
	mmap1.write(test_str, 10);

	mmap2.seek(0, SEEK_SET);
	char buffer[10];
	mmap2.read(buffer, 10);
	EXPECT_EQ(memcmp(buffer, test_str, 10), 0);

	unlink(test_file);
}

TEST(MmapHandleTest, ResizeViaMmap) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);

	struct stat st;
	stat(test_file, &st);
	EXPECT_EQ(st.st_size, 0);

	MmapHandle mmap = handle.getMmapHandle(0, 512);

	stat(test_file, &st);
	EXPECT_EQ(st.st_size, 512);

	unlink(test_file);
}

TEST(MmapHandleTest, MmapWithOffset) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);
	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
	ASSERT_TRUE(handle);
	MmapHandle full_map = handle.getMmapHandle(0, 8192);
	ASSERT_TRUE(full_map);

	const char str1[] = "Start";
	full_map.write(str1, 5);

	MmapHandle offset_map = handle.getMmapHandle(4096, 4096);
	ASSERT_TRUE(offset_map);

	const char str2[] = "Offset";
	offset_map.write(str2, 6);

	full_map.seek(4096, SEEK_SET);
	char buffer[6];
	full_map.read(buffer, 6);
	EXPECT_EQ(memcmp(buffer, str2, 6), 0);

	unlink(test_file);
}

TEST(MmapHandleTest, ReadOnlyMmap) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);
	const char test_content[] = "ReadOnly";
	{
		FdHandle handle_write = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);
		MmapHandle writer = handle_write.getMmapHandle(0, 1024);
		writer.write(test_content, 8);
	}

	FdHandle handle_read = FdHandle::open(test_file, O_RDONLY);
	MmapHandle reader = handle_read.getMmapHandle(0, 1024, PROT_READ, MAP_SHARED);

	EXPECT_STREQ(reader.directPointer<const char>(), test_content);

	unlink(test_file);
}



TEST(MmapHandleTest, MultiThreadMmap_LargeBlockNoContention) {
	const char* test_file = "fd_mmap_test.tmp";
	unlink(test_file);

	const size_t blockSize = 32 * 1024 * 1024;
	const int threadCount = 4;

	FdHandle handle = FdHandle::open(test_file, O_RDWR | O_CREAT, 0660);

	std::vector<MmapHandle> mmaps;
	for (int i = 0; i < threadCount; i++) {
		mmaps.emplace_back(handle.getMmapHandle(i * blockSize, blockSize));
	}

	std::atomic<int> ready{0};
	std::atomic<bool> start{false};

	auto worker = [&](int idx) {
		MmapHandle& mmap = mmaps[idx];

		ready.fetch_add(1, std::memory_order_release);
		while (!start.load(std::memory_order_acquire)) {
			// spin
		}

		mmap.seek(0, SEEK_SET);

		for (size_t i = 0; i < blockSize; i++) {
			uint8_t v = (uint8_t)(91 * i);
			mmap.write(&v, 1);
		}
	};

	std::vector<std::thread> threads;
	for (int i = 0; i < threadCount; i++) {
		threads.emplace_back(worker, i);
	}

	while (ready.load(std::memory_order_acquire) != threadCount) {
		std::this_thread::yield();
	}

	uint64_t t0 = millis_since_epoch();
	start.store(true, std::memory_order_release);

	for (auto& t : threads) {
		t.join();
	}

	uint64_t t1 = millis_since_epoch();
	uint64_t multiMs = t1 - t0;

	// ---- correctness verification ----
	std::vector<uint8_t> verify(blockSize);

	for (int i = 0; i < threadCount; i++) {
		handle.seek(i * blockSize, SEEK_SET);
		handle.read(verify.data(), verify.size());

		for (size_t j = 0; j < blockSize; j++) {
			ASSERT_EQ(verify[j], (uint8_t)(91 * j));
		}
	}

	// ---- single-thread baseline ----
	MmapHandle single = handle.getMmapHandle(threadCount * blockSize, blockSize);

	uint64_t s0 = millis_since_epoch();
	single.seek(0, SEEK_SET);
	for (size_t i = 0; i < blockSize; i++) {
		uint8_t v = (uint8_t)(91 * i);
		single.write(&v, 1);
	}
	uint64_t s1 = millis_since_epoch();

	uint64_t singleMs = s1 - s0;

	EXPECT_LT(multiMs, singleMs * threadCount);

	unlink(test_file);
}

TEST(MmapHandleTest, MoveConstructor) {
	const char* temp_filename = "mmap_test.tmp";
	unlink(temp_filename);
	FdHandle fd_handle = FdHandle::open(temp_filename, O_RDWR | O_CREAT, 0660);
	MmapHandle original = fd_handle.getMmapHandle(0, 1024);
	EXPECT_TRUE(original);

	MmapHandle moved(std::move(original));
	EXPECT_TRUE(moved);
	EXPECT_FALSE(original);

	unlink(temp_filename);
}
