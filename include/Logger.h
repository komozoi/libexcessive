//
// Created by nuclaer on 14.2.25.
//

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
