/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-02-14
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


#ifndef EXCESSIVE_LOGGER_H
#define EXCESSIVE_LOGGER_H

#include <cstdarg>
#include <mutex>


#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARN 2
#define LOG_LEVEL_ERROR 3
#define LOG_LEVEL_CRIT 4


class Logger {
public:
	Logger(const char* dir, int fileLevel, int outLevel);

	void log(int level, const char* process, const char* fmt, va_list arg);

	~Logger();

	static const char* levelNames[5];

private:

	int fd, fileLevel, outLevel;

	std::mutex lock;
};


class LogEndpoint {
public:
	LogEndpoint(Logger& log, const char* process) : log(log), process(process) {}

	void debug(const char* fmt, ...);
	void info(const char* fmt, ...);
	void warning(const char* fmt, ...);
	void error(const char* fmt, ...);
	void critical(const char* fmt, ...);

private:
	Logger& log;
	const char* process;
};


#endif //EXCESSIVE_LOGGER_H
