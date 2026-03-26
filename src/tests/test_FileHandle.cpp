//
// Created by komozoi on 22.2.25.
//

#include "fs/FdHandle.h"
#include <gtest/gtest.h>
#include "fcntl.h"
#include <string>


#define TEST_FILE "test_file.txt"


using namespace std;

class FdHandleTest : public ::testing::Test {
protected:

	// Set up (called before each test)
	void SetUp() override {
		// Clean up any existing test file
		if (remove(TEST_FILE) != 0 && errno != ENOENT) {
			// Ignore errors if the file doesn't exist
		}
	}

	// Tear down (called after each test)
	void TearDown() override {
		// Clean up the test file
		remove(TEST_FILE);
	}
};


// Test case: Opening a new file in write mode
TEST_F(FdHandleTest, OpenWriteMode) {
	// Create a new file
	FdHandle handle = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);

	// Verify that isNew() returns true for a newly created file
	EXPECT_TRUE(handle.isNew());

	// Check that writes work as expected
	EXPECT_EQ(handle.write((uint64_t)0xdeadbeef12345678UL), 8);
}

// Test case: Opening an existing file in read mode
TEST_F(FdHandleTest, OpenReadMode) {

	system("echo \"testtest\" > " TEST_FILE);

	// Open the file in read-only mode
	FdHandle handle = FdHandle::open(TEST_FILE, O_RDONLY);

	// Verify that isNew() returns false for an existing file opened in read mode
	EXPECT_FALSE(handle.isNew());

	uint64_t out;
	EXPECT_EQ(handle.read(out), 8);
}

// Test case: Writing to and reading from a file
TEST_F(FdHandleTest, WriteAndRead) {
	const string test_content = "Hello, World!";

	{
		// Create and write to the file
		FdHandle handle_write = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
		handle_write.write(test_content.data(), test_content.size());
		handle_write.flush();
	}

	{
		// Open the file for reading
		FdHandle handle_read = FdHandle::open(TEST_FILE, O_RDONLY);
		char buffer[1024];
		size_t bytes_read = handle_read.read(buffer, sizeof(buffer));
		ASSERT_EQ(bytes_read, test_content.size());

		// Verify that the content was read correctly
		buffer[bytes_read] = 0;
		EXPECT_STREQ(buffer, test_content.c_str());
	}
}

// Test case: Proper cleanup on deletion
TEST_F(FdHandleTest, CloseOnReferenceLost) {
	int extractedFd;

	{
		FdHandle handle = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
		extractedFd = handle.getFd();

		// Do random stuff, checking that file actually opened
		uint64_t val = 15;
		ASSERT_EQ(handle.write(val), 8);
		EXPECT_NE(fcntl(extractedFd, F_GETFD), -1);
	}

	// Check that the file was closed
	EXPECT_EQ(fcntl(extractedFd, F_GETFD), -1);
	EXPECT_EQ(errno, EBADF);
}

TEST_F(FdHandleTest, FdHandleSize) {
	EXPECT_EQ(sizeof(FdHandle), 2);
}

/*
TEST_F(FdHandleTest, WriteAndReadMmap) {
	const string test_content = "Hello, World!";

	{
		// Create and write to the file
		FdHandle handle_write = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
		MmapHandle writer = handle_write.getMmapHandle(0, 1024);
		writer.write(test_content.data(), test_content.size());
		writer.write<uint8_t>(0);
	}

	{
		// Open the file for reading
		FdHandle handle_read = FdHandle::open(TEST_FILE, O_RDONLY);
		MmapHandle reader = handle_read.getMmapHandle(0, 1024);

		// Verify that the content was read correctly
		EXPECT_STREQ(reader.directPointer<const char>(), test_content.c_str());
	}
}

TEST_F(FdHandleTest, CloseOnReferenceLostMmap) {
	FdHandle handle = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	{
		MmapHandle writer = handle.getMmapHandle(0, 1024);
		writer.write<int>(0);
		EXPECT_EQ(handle.numReferences(), 2);
	}

	EXPECT_EQ(handle.numReferences(), 1);
}*/
