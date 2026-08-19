/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-4-9
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

#include "parallel/ThreadPool.h"
#include <gtest/gtest.h>
#include "ds/ArrayList.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>

TEST(ThreadPoolTest, PoolSize) {
	ThreadPool pool(4);
	EXPECT_EQ(pool.getPoolSize(), 4);
}

TEST(ThreadPoolTest, BasicExecution) {
	ThreadPool pool(4);
	std::atomic<int> result(0);
	std::atomic<bool> done(false);
	
	pool.submit([&result, &done]() {
		result = 10 * 10;
		done = true;
	});
	
	while (!done) {
		std::this_thread::yield();
	}
	EXPECT_EQ(result.load(), 100);
}

TEST(ThreadPoolTest, MultipleTasks) {
	ThreadPool pool(4);
	const int numTasks = 10;
	ArrayList<int> results;
	for (int i = 0; i < numTasks; ++i) {
		results.add(0);
	}
	std::atomic<int> completed(0);
	
	for (int i = 0; i < numTasks; ++i) {
		pool.submit([i, &results, &completed]() {
			results.get(i) = i;
			completed++;
		});
	}
	
	while (completed < numTasks) {
		std::this_thread::yield();
	}
	
	for (int i = 0; i < numTasks; ++i) {
		EXPECT_EQ(results.get(i), i);
	}
}

TEST(ThreadPoolTest, Shutdown) {
	ThreadPool pool(4);
	EXPECT_FALSE(pool.isShutdown());
	pool.shutdown();
	EXPECT_TRUE(pool.isShutdown());
}

TEST(ThreadPoolTest, SubmitAfterShutdown) {
	ThreadPool pool(4);
	pool.shutdown();
	EXPECT_THROW(pool.submit([]() {}), std::runtime_error);
}

TEST(ThreadPoolTest, QueueSize) {
	// Block threads to fill the queue
	ThreadPool pool(1);
	std::mutex m;
	std::unique_lock<std::mutex> lock(m);
	std::atomic<bool> taskStarted(false);
	
	pool.submit([&m, &taskStarted]() {
		taskStarted = true;
		std::unique_lock<std::mutex> innerLock(m);
	});
	
	// Wait for the task to actually start and block
	while (!taskStarted) {
		std::this_thread::yield();
	}
	
	// Submit more tasks
	pool.submit([]() {});
	pool.submit([]() {});
	
	EXPECT_EQ(pool.getQueueSize(), 2);
	lock.unlock();
}

TEST(ThreadPoolTest, LargeNumberOfTasks) {
	ThreadPool pool(8);
	const int numTasks = 1000;
	std::atomic<int> counter(0);
	
	for (int i = 0; i < numTasks; ++i) {
		pool.submit([&counter]() { counter++; });
	}
	
	pool.shutdown(); // Shutdown waits for all tasks
	EXPECT_EQ(counter.load(), numTasks);
}

TEST(ThreadPoolTest, VoidTask) {
	ThreadPool pool(2);
	std::atomic<bool> flag(false);
	pool.submit([&flag]() { flag = true; });
	
	while (!flag) {
		std::this_thread::yield();
	}
	EXPECT_TRUE(flag);
}

TEST(ThreadPoolTest, TaskWithArguments) {
	ThreadPool pool(2);
	std::atomic<int> result(0);
	std::atomic<bool> done(false);
	int a = 5;
	int b = 7;
	
	pool.submit([a, b, &result, &done]() {
		result = a + b;
		done = true;
	});
	
	while (!done) {
		std::this_thread::yield();
	}
	EXPECT_EQ(result.load(), 12);
}

TEST(ThreadPoolTest, DestructorShutsDown) {
	std::atomic<int> counter(0);
	{
		ThreadPool pool(4);
		for (int i = 0; i < 10; ++i) {
			pool.submit([&counter]() {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				counter++;
			});
		}
	}
	// Pool is destroyed, and its destructor calls shutdown(), which joins all threads.
	// So all submitted tasks should have finished.
	EXPECT_EQ(counter.load(), 10);
}

TEST(ThreadPoolTest, MultipleShutdowns) {
	ThreadPool pool(2);
	pool.shutdown();
	pool.shutdown();
	EXPECT_TRUE(pool.isShutdown());
}

TEST(ThreadPoolTest, BusyPool) {
	ThreadPool pool(2);
	std::atomic<int> running(0);
	std::atomic<int> total(0);
	const int numTasks = 20;

	for (int i = 0; i < numTasks; ++i) {
		pool.submit([&running, &total]() {
			running++;
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			total++;
			running--;
		});
	}

	pool.shutdown();
	EXPECT_EQ(total.load(), numTasks);
	EXPECT_EQ(running.load(), 0);
}

void testFunc(int a, int b, std::atomic<int>& result) {
	result = a + b;
}

TEST(ThreadPoolTest, SubmitWithMultipleArguments) {
	ThreadPool pool(2);
	std::atomic<int> result(0);
	sp<ThreadPoolTask> task = pool.submit(testFunc, 10, 20, std::ref(result));

	while (!task->isDone()) {
		std::this_thread::yield();
	}
	EXPECT_EQ(result.load(), 30);
}

TEST(ThreadPoolTest, Wait_BlocksUntilTaskFinishes) {
	ThreadPool pool(1);
	std::mutex block;
	std::unique_lock<std::mutex> hold(block);
	std::atomic<bool> started(false);
	std::atomic<bool> finished(false);

	sp<ThreadPoolTask> task = pool.submit([&block, &started, &finished]() {
		started.store(true);
		std::lock_guard<std::mutex> inner(block);
		finished.store(true);
	});

	while (!started.load()) {
		std::this_thread::yield();
	}

	std::atomic<bool> waiterReturned(false);
	std::thread waiter([&task, &waiterReturned]() {
		task->wait();
		waiterReturned.store(true);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	EXPECT_FALSE(waiterReturned.load());
	EXPECT_FALSE(finished.load());

	hold.unlock();
	waiter.join();
	EXPECT_TRUE(finished.load());
	EXPECT_TRUE(task->isDone());
}

TEST(ThreadPoolTest, WaitAll_SeesAllSideEffects) {
	ThreadPool pool(4);
	const int numTasks = 8;
	std::atomic<int> counter(0);
	ArrayList<sp<ThreadPoolTask>> tasks;
	for (int i = 0; i < numTasks; ++i) {
		tasks.add(pool.submit([&counter]() {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			counter.fetch_add(1);
		}));
	}
	ThreadPool::waitAll(tasks);
	EXPECT_EQ(counter.load(), numTasks);
	for (int i = 0; i < tasks.size(); ++i) {
		EXPECT_TRUE(tasks.get(i)->isDone());
	}
}

struct PForCtx {
	std::atomic<int> hits[16];
	std::atomic<int> nWorkers;
	std::atomic<int> bad;
	ThreadPool* pool;
};

static void pforRecord(int worker, int nWorkers, void* raw) {
	PForCtx* c = (PForCtx*)raw;
	c->nWorkers.store(nWorkers);
	if (worker < 0 || worker >= nWorkers || worker >= 16) {
		c->bad.fetch_add(1);
		return;
	}
	if (c->hits[worker].fetch_add(1) != 0)
		c->bad.fetch_add(1);
}

static void pforNested(int worker, int nWorkers, void* raw) {
	pforRecord(worker, nWorkers, raw);
	PForCtx* c = (PForCtx*)raw;
	if (worker == 0 && c->pool) {
		EXPECT_THROW(c->pool->parallelFor(pforRecord, c), std::runtime_error);
	}
}

TEST(ThreadPoolTest, ParallelFor_UniqueWorkers) {
	ThreadPool pool(4);
	PForCtx ctx{};
	ctx.nWorkers.store(0);
	ctx.bad.store(0);
	for (int i = 0; i < 16; ++i)
		ctx.hits[i].store(0);
	pool.parallelFor(pforRecord, &ctx);
	EXPECT_EQ(ctx.nWorkers.load(), 5);
	EXPECT_EQ(ctx.bad.load(), 0);
	for (int i = 0; i < 5; ++i)
		EXPECT_EQ(ctx.hits[i].load(), 1) << i;
	for (int i = 5; i < 16; ++i)
		EXPECT_EQ(ctx.hits[i].load(), 0);
}

TEST(ThreadPoolTest, ParallelFor_EmptyPool_Inline) {
	ThreadPool pool(0);
	PForCtx ctx{};
	ctx.nWorkers.store(0);
	ctx.bad.store(0);
	for (int i = 0; i < 16; ++i)
		ctx.hits[i].store(0);
	pool.parallelFor(pforRecord, &ctx);
	EXPECT_EQ(ctx.nWorkers.load(), 1);
	EXPECT_EQ(ctx.hits[0].load(), 1);
	EXPECT_EQ(ctx.bad.load(), 0);
}

TEST(ThreadPoolTest, ParallelFor_NotNestable) {
	ThreadPool pool(2);
	PForCtx ctx{};
	ctx.nWorkers.store(0);
	ctx.bad.store(0);
	ctx.pool = &pool;
	for (int i = 0; i < 16; ++i)
		ctx.hits[i].store(0);
	pool.parallelFor(pforNested, &ctx);
	EXPECT_EQ(ctx.bad.load(), 0);
	EXPECT_EQ(ctx.nWorkers.load(), 3);
}

TEST(ThreadPoolTest, ParallelFor_AfterShutdownThrows) {
	ThreadPool pool(2);
	pool.shutdown();
	PForCtx ctx{};
	EXPECT_THROW(pool.parallelFor(pforRecord, &ctx), std::runtime_error);
}

TEST(ThreadPoolTest, DefaultPool_SameInstanceAndNproc) {
	ThreadPool& a = ThreadPool::getDefault();
	ThreadPool& b = ThreadPool::getDefault();
	EXPECT_EQ(&a, &b);
	unsigned hc = std::thread::hardware_concurrency();
	int expect = (hc < 1) ? 1 : (int)hc;
	EXPECT_EQ(a.getPoolSize(), expect);
	EXPECT_FALSE(a.isShutdown());
}

TEST(ThreadPoolTest, DefaultPool_RunsWork) {
	ThreadPool& pool = ThreadPool::getDefault();
	std::atomic<int> result(0);
	sp<ThreadPoolTask> task = pool.submit([&result]() {
		result.store(42);
	});
	task->wait();
	EXPECT_EQ(result.load(), 42);
}
