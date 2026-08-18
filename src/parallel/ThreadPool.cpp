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
#include <stdexcept>

ThreadPool::ThreadPool(int threads) : stop(false) {
	for (int i = 0; i < threads; ++i) {
		workers.add(std::thread(&ThreadPool::workerLoop, this, i + 1));
	}
}

ThreadPool::~ThreadPool() {
	shutdown();
}

void ThreadPool::workerLoop(int workerId) {
	unsigned lastEpoch = 0;
	for (;;) {
		sp<ThreadPoolTask> task;
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			for (;;) {
				const unsigned ep = pforEpoch.load(std::memory_order_relaxed);
				if (ep != lastEpoch)
					break;
				if (!tasks.empty())
					break;
				if (stop)
					return;
				condition.wait(lock);
			}
			const unsigned ep = pforEpoch.load(std::memory_order_relaxed);
			if (ep != lastEpoch) {
				lastEpoch = ep;
				ForFn fn = pforFn;
				void* ctx = pforCtx;
				int nw = pforNWorkers;
				lock.unlock();
				if (fn)
					fn(workerId, nw, ctx);
				if (pforDone.fetch_add(1, std::memory_order_acq_rel) + 1 >= nw - 1)
					condition.notify_all();
				continue;
			}
			if (stop && tasks.empty())
				return;
			task = tasks.pop();
		}
		try {
			task.mut().run();
		} catch (...) {
			task.mut().markFinished();
			throw;
		}
		task.mut().markFinished();
	}
}

void ThreadPool::parallelFor(ForFn fn, void* ctx) {
	if (!fn)
		throw std::invalid_argument("ThreadPool::parallelFor: null fn");
	const int nPool = workers.size();
	if (nPool == 0) {
		fn(0, 1, ctx);
		return;
	}

	{
		std::lock_guard<std::mutex> lock(queueMutex);
		if (stop)
			throw std::runtime_error("parallelFor on stopped ThreadPool");
		if (pforBusy)
			throw std::runtime_error("ThreadPool::parallelFor is not nestable");
		pforBusy = true;
		pforFn = fn;
		pforCtx = ctx;
		pforNWorkers = nPool + 1;
		pforDone.store(0, std::memory_order_relaxed);
		pforEpoch.fetch_add(1, std::memory_order_release);
	}
	condition.notify_all();
	fn(0, nPool + 1, ctx);
	{
		std::unique_lock<std::mutex> lock(queueMutex);
		while (pforDone.load(std::memory_order_acquire) < nPool)
			condition.wait(lock);
		pforBusy = false;
		pforFn = nullptr;
		pforCtx = nullptr;
	}
}

void ThreadPool::submit(sp<ThreadPoolTask> task) {
	{
		std::lock_guard<std::mutex> lock(queueMutex);

		if (stop) {
			throw std::runtime_error("submit on stopped ThreadPool");
		}

		tasks.add(task);
	}
	condition.notify_one();
}

void ThreadPool::shutdown() {
	{
		std::lock_guard<std::mutex> lock(queueMutex);
		if (stop) {
			return;
		}
		stop = true;
	}
	condition.notify_all();
	for (int i = 0; i < workers.size(); ++i) {
		if (workers.get(i).joinable()) {
			workers.get(i).join();
		}
	}
}

bool ThreadPool::isShutdown() const {
	std::lock_guard<std::mutex> lock(queueMutex);
	return stop;
}

int ThreadPool::getPoolSize() const {
	return workers.size();
}

int ThreadPool::getQueueSize() const {
	std::lock_guard<std::mutex> lock(queueMutex);
	return tasks.size();
}

bool ThreadPoolTask::isDone() const {
	return finished.load(std::memory_order_acquire);
}

void ThreadPoolTask::markFinished() {
	{
		std::lock_guard<std::mutex> lock(doneMutex);
		finished.store(true, std::memory_order_release);
	}
	doneCv.notify_all();
}

void ThreadPoolTask::wait() const {
	if (finished.load(std::memory_order_acquire))
		return;
	std::unique_lock<std::mutex> lock(doneMutex);
	while (!finished.load(std::memory_order_acquire))
		doneCv.wait(lock);
}

void ThreadPool::waitAll(const ArrayList<sp<ThreadPoolTask>>& tasks) {
	for (int i = 0; i < tasks.size(); ++i) {
		const sp<ThreadPoolTask>& task = tasks.get(i);
		if (task)
			task->wait();
	}
}

static int defaultWorkerCount() {
	unsigned hc = std::thread::hardware_concurrency();
	if (hc < 1)
		return 1;
	return (int)hc;
}

ThreadPool& ThreadPool::getDefault() {
	static ThreadPool pool(defaultWorkerCount());
	return pool;
}
