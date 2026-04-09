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

#include "ds/Queue.h"
#include "ds/ArrayList.h"
#include <gtest/gtest.h>

template<class T>
class QueueTest : public ::testing::Test {
protected:
	void SetUp() override {
		// Initialize queues for testing
		q1 = new Queue<T>();
		q2 = new Queue<T>();
	}

	void TearDown() override {
		delete q1;
		delete q2;
	}

	Queue<T>* q1;
	Queue<T>* q2;
};

// Test cases for int type
typedef QueueTest<int> IntQueueTest;

TEST_F(IntQueueTest, BasicAddAndPop) {
	// Add elements and check size
	q1->add(1);
	q1->add(2);
	EXPECT_EQ(q1->size(), 2);

	// Pop elements in order
	int popped = q1->pop();
	EXPECT_EQ(popped, 1);
	EXPECT_EQ(q1->size(), 1);

	popped = q1->pop();
	EXPECT_EQ(popped, 2);
	EXPECT_TRUE(q1->empty());
}

TEST_F(IntQueueTest, EmptyAndSize) {
	// Check empty and size for an empty queue
	EXPECT_TRUE(q1->empty());
	EXPECT_EQ(q1->size(), 0);

	// Add elements and check again
	q1->add(5);
	EXPECT_FALSE(q1->empty());
	EXPECT_EQ(q1->size(), 1);

	// Pop and check
	q1->pop();
	EXPECT_TRUE(q1->empty());
}

TEST_F(IntQueueTest, CopyConstructor) {
	// Add elements to original queue
	q1->add(3);
	q1->add(4);

	// Create a copy
	Queue<int> qCopy(*q1);

	// Modify the original queue
	q1->add(5);

	// Check that the copy has only the initial elements
	EXPECT_EQ(qCopy.size(), 2);
	int popped = qCopy.pop();
	EXPECT_EQ(popped, 3);
}

TEST_F(IntQueueTest, MoveConstructor) {
	// Add elements to original queue
	q1->add(6);
	q1->add(7);

	// Move the queue
	Queue<int> qMoved(std::move(*q1));

	// The original should be empty after move
	EXPECT_TRUE(q1->empty());

	// Check that moved queue has the elements
	EXPECT_EQ(qMoved.size(), 2);
}

TEST_F(IntQueueTest, AssignmentOperator) {
	// Add elements to original queue
	q1->add(8);
	q1->add(9);

	// Assign to another queue
	*q2 = *q1;

	// Modify the original queue
	q1->add(10);

	// Check that assigned queue has only initial elements
	EXPECT_EQ(q2->size(), 2);
}

TEST_F(IntQueueTest, RvalueAdd) {
	// Add an rvalue element
	int val = 11;
	q1->add(std::move(val));

	// Pop and check the value
	EXPECT_EQ(q1->pop(), 11);
}

TEST_F(IntQueueTest, Iteration) {
	q1->add(1);
	q1->add(2);
	q1->add(3);

	int sum = 0;
	int count = 0;
	for (int val : *q1) {
		sum += val;
		count++;
	}

	EXPECT_EQ(sum, 6);
	EXPECT_EQ(count, 3);
}

TEST_F(IntQueueTest, ContainerAPI) {
	q1->add(10);
	q1->add(20);

	EXPECT_FALSE(q1->isEmpty());
	EXPECT_EQ(q1->getElement(0), 10);
	EXPECT_EQ(q1->getElement(1), 20);
	EXPECT_EQ(q1->find(20), 1);
	EXPECT_EQ(q1->find(30), -1);

	q1->clear();
	EXPECT_TRUE(q1->isEmpty());
	EXPECT_EQ(q1->size(), 0);
}

TEST_F(IntQueueTest, AddManyFromArrayList) {
	ArrayList<int> list{100, 200};
	q1->addMany(list);
	EXPECT_EQ(q1->size(), 2);
	EXPECT_EQ(q1->pop(), 100);
	EXPECT_EQ(q1->pop(), 200);
}

// Test cases for std::string type (similar structure)
typedef QueueTest<std::string> stringQueueTest;

TEST_F(stringQueueTest, BasicAddAndPop) {
	q1->add("Hello");
	q1->add("World");

	EXPECT_EQ(q1->size(), 2);

	std::string popped = q1->pop();
	EXPECT_EQ(popped, "Hello");
	popped = q1->pop();
	EXPECT_EQ(popped, "World");
}

TEST_F(stringQueueTest, CopyQueue) {
	q1->add("Hello");
	q1->add("World");

	Queue<std::string> q2 = *q1;
	q1->clear();

	EXPECT_EQ(q2.size(), 2);

	std::string popped = q2.pop();
	EXPECT_EQ(popped, "Hello");
	popped = q2.pop();
	EXPECT_EQ(popped, "World");
}

