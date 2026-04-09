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

#ifndef EXCESSIVE_THREADPOOL_H
#define EXCESSIVE_THREADPOOL_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "alloc/pointer.h"
#include "ds/ArrayList.h"
#include "ds/Queue.h"


class ThreadPoolTask {
public:
	virtual void run() = 0;
	virtual bool isDone() const = 0;

	virtual ~ThreadPoolTask() = default;
};


class ThreadPoolTaskFunction : public ThreadPoolTask {
public:
	template<typename Func, typename = typename std::enable_if<!std::is_same<typename std::decay<Func>::type, ThreadPoolTaskFunction>::value>::type>
	ThreadPoolTaskFunction(Func&& func)
		: task(std::forward<Func>(func)) {}

	void operator()() {
		run();
	}

	void run() override {
		task();
		didRun = true;
	}

	bool isDone() const override {
		return didRun;
	}

private:
	std::function<void()> task;
	bool didRun = false;
};


/**
 * @brief A thread pool for executing tasks in parallel.
 *
 * This class manages a pool of threads and a queue of tasks.
 */
class ThreadPool {
public:
	/**
	 * @brief Constructs a new ThreadPool with the specified number of threads.
	 * @param threads The number of threads in the pool.
	 */
	explicit ThreadPool(int threads);

	/**
	 * @brief Destructor. Shuts down the thread pool.
	 */
	~ThreadPool();

	/**
	 * @brief Submits a task to the thread pool for execution.
	 * @param task The task to execute.
	 */
	void submit(sp<ThreadPoolTask> task);

	/**
	 * @brief Submits a task to the thread pool for execution.
	 * @param func The function to execute.
	 */
	template<typename Func, typename = typename std::enable_if<!std::is_base_of<ThreadPoolTask, typename std::remove_reference<Func>::type>::value &&
	                                                             !is_sp<typename std::remove_reference<Func>::type>::value>::type>
	sp<ThreadPoolTask> submit(Func&& func) {
		sp<ThreadPoolTask> task = sp<ThreadPoolTaskFunction>::create(std::forward<Func>(func));
		submit(task);
		return task;
	}

	/**
	 * @brief Submits a task to the thread pool for execution with arguments.
	 * @param func The function to execute.
	 * @param args The arguments to pass to the function.
	 */
	template<typename Func, typename... Args, typename = typename std::enable_if<(sizeof...(Args) > 0)>::type>
	sp<ThreadPoolTask> submit(Func&& func, Args&&... args) {
		sp<ThreadPoolTask> task = sp<ThreadPoolTaskFunction>::create(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));
		submit(task);
		return task;
	}

	/**
	 * @brief Shuts down the thread pool gracefully.
	 */
	void shutdown();

	/**
	 * @brief Checks if the thread pool has been shut down.
	 * @return True if the thread pool is shut down, false otherwise.
	 */
	bool isShutdown() const;

	/**
	 * @brief Gets the number of threads in the pool.
	 * @return The number of threads.
	 */
	int getPoolSize() const;

	/**
	 * @brief Gets the number of tasks currently in the queue.
	 * @return The number of tasks in the queue.
	 */
	int getQueueSize() const;

private:
	void workerLoop();

	ArrayList<std::thread> workers;
	Queue<sp<ThreadPoolTask>> tasks;

	mutable std::mutex queueMutex;
	std::condition_variable condition;
	bool stop;
};

#endif // EXCESSIVE_THREADPOOL_H
