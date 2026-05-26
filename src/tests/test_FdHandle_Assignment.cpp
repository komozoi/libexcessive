/*
 * Copyright 2023-2025 komozoi
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
#include <gtest/gtest.h>
#include "fcntl.h"
#include <unistd.h>

#define TEST_FILE "test_assignment.txt"

class FdHandleAssignmentTest : public ::testing::Test {
protected:
	void SetUp() override {
		remove(TEST_FILE);
		remove("another_test.txt");
	}

	void TearDown() override {
		remove(TEST_FILE);
		remove("another_test.txt");
	}
};

TEST_F(FdHandleAssignmentTest, NormalAssignment) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	ASSERT_TRUE(h1);
	int fd1 = h1.getFd();
	EXPECT_EQ(h1.numReferences(), 1);

	FdHandle h2;
	EXPECT_FALSE(h2);

	h2 = h1;
	EXPECT_TRUE(h2);
	EXPECT_EQ(h2.getFd(), fd1);
	EXPECT_EQ(h1.numReferences(), 2);
	EXPECT_EQ(h2.numReferences(), 2);
}

TEST_F(FdHandleAssignmentTest, SelfAssignment) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	int initialRefs = h1.numReferences();
	int fd1 = h1.getFd();

	h1 = h1;

	EXPECT_TRUE(h1);
	EXPECT_EQ(h1.getFd(), fd1);
	EXPECT_EQ(h1.numReferences(), initialRefs);
}

TEST_F(FdHandleAssignmentTest, ReassignmentReleasesOld) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	int fd1 = h1.getFd();

	FdHandle h2 = FdHandle::open("another_test.txt", O_WRONLY | O_CREAT, 0660);
	int fd2 = h2.getFd();

	h1 = h2;

	EXPECT_EQ(h1.getFd(), fd2);
	EXPECT_EQ(h1.numReferences(), 2);
	EXPECT_EQ(h2.numReferences(), 2);

	// fd1 should be closed because h1 was the only handle
	EXPECT_EQ(fcntl(fd1, F_GETFD), -1);
	EXPECT_EQ(errno, EBADF);
}

TEST_F(FdHandleAssignmentTest, AssignmentFromInvalid) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	int fd1 = h1.getFd();
	FdHandle h2; // invalid, fd = -1

	h1 = h2;

	EXPECT_FALSE(h1);
	EXPECT_EQ(h1.getFd(), -1);
	
	// fd1 should be closed
	EXPECT_EQ(fcntl(fd1, F_GETFD), -1);
	EXPECT_EQ(errno, EBADF);
}

TEST_F(FdHandleAssignmentTest, AssignmentToInvalid) {
	FdHandle h1;
	FdHandle h2 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	int fd2 = h2.getFd();

	h1 = h2;

	EXPECT_TRUE(h1);
	EXPECT_EQ(h1.getFd(), fd2);
	EXPECT_EQ(h2.numReferences(), 2);
}

TEST_F(FdHandleAssignmentTest, ChainAssignment) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	FdHandle h2, h3;

	h3 = h2 = h1;

	EXPECT_EQ(h1.numReferences(), 3);
	EXPECT_EQ(h1.getFd(), h2.getFd());
	EXPECT_EQ(h1.getFd(), h3.getFd());
}

TEST_F(FdHandleAssignmentTest, SameFdDifferentHandleObjects) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	int fd = h1.getFd();
	FdHandle h2 = FdHandle::from(fd);
	
	EXPECT_EQ(h1.numReferences(), 2);

	h1 = h2;

	EXPECT_EQ(h1.getFd(), fd);
	EXPECT_EQ(h1.numReferences(), 2);
}

TEST_F(FdHandleAssignmentTest, MultipleAssignmentSameHandle) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	FdHandle h2 = FdHandle::open("another_test.txt", O_WRONLY | O_CREAT, 0660);
	
	int fd2 = h2.getFd();

	h1 = h2;
	h1 = h2;
	h1 = h2;

	EXPECT_EQ(h1.getFd(), fd2);
	EXPECT_EQ(h1.numReferences(), 2);
}

TEST_F(FdHandleAssignmentTest, AssignmentFromSelfViaAlias) {
	FdHandle h1 = FdHandle::open(TEST_FILE, O_WRONLY | O_CREAT, 0660);
	FdHandle& alias = h1;
	int initialRefs = h1.numReferences();

	h1 = alias;

	EXPECT_EQ(h1.numReferences(), initialRefs);
}

TEST_F(FdHandleAssignmentTest, InvalidHandleRefcountGrowth) {
	FdHandle h1;
	int ref1 = h1.numReferences();
	
	FdHandle h2;
	h1 = h2;
	
	int ref2 = h1.numReferences();
	EXPECT_EQ(ref1, ref2) << "Refcount of invalid handle should not grow on assignment";
}
