/*
 * Copyright 2023-2026 komozoi
 * Original Creation Date: 2025-10-13
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
 *
 */

#include "gtest/gtest.h"
#include "fs/DiskBytestringSearchTree.h"
#include "fs/FdHandle.h"

#include "fcntl.h"


#define TEST_FILE_PATH "/tmp/DiskBytestringSearchTreeTest_"



TEST(DiskBytestringSearchTreeTest, InsertSingleKeyAndFindIt) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertSingleKeyAndFindIt.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	Bytestring key = "hello";
	uint64_t value = 1234;

	ASSERT_FALSE(tree.insert(key, value));
	ASSERT_EQ(tree.find(key), value);
}

TEST(DiskBytestringSearchTreeTest, FindMissingKeyReturnsZero) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "FindMissingKeyReturnsZero.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	ASSERT_EQ(tree.find("nonexistent"), 0);
}

TEST(DiskBytestringSearchTreeTest, InsertDuplicateKeyDoesNotOverwrite) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertDuplicateKeyDoesNotOverwrite.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	Bytestring key = "dup";
	uint64_t val1 = 1111;
	uint64_t val2 = 2222;

	ASSERT_FALSE(tree.insert(key, val1));
	ASSERT_TRUE(tree.insert(key, val2));
	ASSERT_EQ(tree.find(key), val1);
}

TEST(DiskBytestringSearchTreeTest, InsertDuplicateKeyOverwrite) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertDuplicateKeyOverwrite.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	Bytestring key = "dup";
	uint64_t val1 = 1111;
	uint64_t val2 = 2222;

	ASSERT_FALSE(tree.insert(key, val1));
	ASSERT_TRUE(tree.insert(key, val2, true));
	ASSERT_EQ(tree.find(key), val2);
}

TEST(DiskBytestringSearchTreeTest, InsertPrefixOfExistingKey) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertPrefixOfExistingKey.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	ASSERT_FALSE(tree.insert("prefix_key", 9001));
	ASSERT_FALSE(tree.insert("prefix", 42));

	ASSERT_EQ(tree.find("prefix"), 42);
	ASSERT_EQ(tree.find("prefix_key"), 9001);
}

TEST(DiskBytestringSearchTreeTest, InsertExtensionOfExistingKey) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertExtensionOfExistingKey.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	ASSERT_FALSE(tree.insert("ext", 1));
	ASSERT_FALSE(tree.insert("ext-123456", 2));

	ASSERT_EQ(tree.find("ext"), 1);
	ASSERT_EQ(tree.find("ext-123456"), 2);
}

TEST(DiskBytestringSearchTreeTest, DivergesCorrectlyAtFragmentBoundary) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "DivergesCorrectlyAtFragmentBoundary.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	ASSERT_FALSE(tree.insert("abcde", 100));
	ASSERT_FALSE(tree.insert("abcdn", 200));
	ASSERT_FALSE(tree.insert("abcdz", 300));

	EXPECT_EQ(tree.find("abcde"), 100);
	EXPECT_EQ(tree.find("abcdn"), 200);
	EXPECT_EQ(tree.find("abcdz"), 300);
}

TEST(DiskBytestringSearchTreeTest, InsertManySmallKeys) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertManySmallKeys.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	for (int i = 0; i < 3000; ++i) {
		Bytestring key(&i, 4);
		ASSERT_FALSE(tree.insert(key, i + 1000));
	}

	for (int i = 0; i < 3000; ++i) {
		Bytestring key(&i, 4);
		EXPECT_EQ(tree.find(key), i + 1000);
	}
}

TEST(DiskBytestringSearchTreeTest, InsertLongKey) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertLongKey.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	std::string longKey(5000, 'x');
	Bytestring key((void*)longKey.data(), longKey.size());

	ASSERT_FALSE(tree.insert(key, 42));
	ASSERT_EQ(tree.find(key), 42);
}

TEST(DiskBytestringSearchTreeTest, InsertBinaryKey) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "InsertBinaryKey.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	uint8_t raw[] = {0x00, 0xFF, 0x10, 0x20, 0x01};
	Bytestring key(raw, sizeof(raw));

	ASSERT_FALSE(tree.insert(key, 9999));
	ASSERT_EQ(tree.find(key), 9999);
}

TEST(DiskBytestringSearchTreeTest, PersistenceAcrossRestart) {
	Bytestring key = "0x1371";
	uint64_t value = 0x9201;

	{
		FdHandle handle = FdHandle::open(TEST_FILE_PATH "PersistenceAcrossRestart.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
		FreeSpaceFile fss(std::move(handle));
		DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));
		ASSERT_FALSE(tree.insert(key, value));
	}

	{
		FdHandle handle = FdHandle::open(TEST_FILE_PATH "PersistenceAcrossRestart.bin", O_RDWR, 0644);
		FreeSpaceFile fss(std::move(handle));
		DiskBytestringSearchTree tree(fss, fss.getHeaderEnd());
		ASSERT_EQ(tree.find(key), value);
	}
}

TEST(DiskBytestringSearchTreeTest, RemoveKeyShouldDeleteIt) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "RemoveKeyShouldDeleteIt.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	Bytestring key = "remove-me";
	uint64_t value = 555;

	ASSERT_FALSE(tree.insert(key, value));
	ASSERT_EQ(tree.remove(key), value);
	ASSERT_EQ(tree.find(key), 0);
}

TEST(DiskBytestringSearchTreeTest, RemoveNonexistentKeyReturnsFalse) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "RemoveNonexistentKeyReturnsFalse.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	ASSERT_EQ(tree.remove("nonexistent"), 0);
}

TEST(DiskBytestringSearchTreeTest, FindReturnsZeroWhenKeyMissing) {
	FdHandle handle = FdHandle::open(TEST_FILE_PATH "FindReturnsZeroWhenKeyMissing.bin", O_RDWR | O_CREAT | O_TRUNC, 0644);
	FreeSpaceFile fss(std::move(handle));
	DiskBytestringSearchTree tree(fss, DiskBytestringSearchTree::initialize(fss));

	ASSERT_EQ(tree.find("ghost-key"), 0);
}


