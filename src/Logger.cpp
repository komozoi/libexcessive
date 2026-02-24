//
// Created by nuclaer on 14.2.25.
//

#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctime>
#include "Logger.h"
#include "unistd.h"
#include "strutil.h"


#define TIMESTAMP_FMT "%Y-%m-%d %H:%M:%S : "
#define TIMESTAMP_LEN 22
#define GRAMMAR_FMT "[%s] %-12s | "
#define GRAMMAR_LEN 14
#define LOG_START_FMT "[  start ] ======== Log started.  Log level: %s ========\n"
#define LOG_CLOSE_FMT "[  stop  ] ======== Log closed. ========\n"


const char* Logger::levelNames[] = {
	"  debug ",
	"  info  ",
	"  warn  ",
	"  error ",
	"critical"
};


Logger::Logger(const char* dir, int fileLevel, int outLevel) : fileLevel(fileLevel), outLevel(outLevel) {
	// Create directory if it doesn't exist
	struct stat st;
	if (stat(dir, &st) != 0) {
		// Directory does not exist, create it
		if (mkdir(dir, S_IRWXU | S_IRWXG | S_IROTH) != 0) {
			// Ideally something should be done here
		}
	}

	// Generate filename based on current date and time
	time_t now = time(0);
	struct tm* localTime = localtime(&now);
	char filePath[strlen(dir) + 32];
	strcpy(filePath, dir);
	strftime(&filePath[strlen(filePath)], 32, "/%Y-%m-%d.txt", localTime);

	// Open the log file
	fd = open(filePath, O_WRONLY | O_CREAT | O_APPEND, 0600);
	if (fd == -1) {
		// Do something??
	} else {
		char buffer[TIMESTAMP_LEN + 5];
		buffer[0] = '\n';
		strftime(buffer + 1, 32, TIMESTAMP_FMT, localTime);
		write(fd, buffer, strlen(buffer));
		dprintf(fd, LOG_START_FMT, levelNames[fileLevel]);
	}
}

void Logger::log(int level, const char* process, const char* fmt, va_list arg) {
	if (level < outLevel && level < fileLevel)
		return;

	time_t now = time(0);
	struct tm* localTime = localtime(&now);

	int contextLen = strlen(process);
	if (contextLen < 12)
		contextLen = 12;
	contextLen += TIMESTAMP_LEN + GRAMMAR_LEN + 8;

	char buffer[contextLen];
	strftime(buffer, 32, TIMESTAMP_FMT, localTime);
	contextLen = concatf(buffer, GRAMMAR_FMT, levelNames[level], process) - buffer;

	// Thread safety
	std::lock_guard<std::mutex> guard(lock);

	if (level >= outLevel) {
		int oFd = level >= LOG_LEVEL_ERROR ? 2 : 1;
		write(oFd, buffer, contextLen);
		va_list tmp;
		va_copy(tmp, arg);
		vdprintf(oFd, fmt, tmp);
		write(oFd, "\n", 1);
		syncfs(oFd);
	}

	if (level >= fileLevel) {
		write(fd, buffer, contextLen);
		vdprintf(fd, fmt, arg);
		write(fd, "\n", 1);
		syncfs(fd);
	}
}

Logger::~Logger() {
	if (fd > 2) {
		time_t now = time(0);
		struct tm* localTime = localtime(&now);

		char buffer[TIMESTAMP_LEN + 4];
		strftime(buffer, 32, TIMESTAMP_FMT, localTime);
		write(fd, buffer, strlen(buffer));
		dprintf(fd, LOG_CLOSE_FMT, levelNames[fileLevel]);

		close(fd);
	}
}

void LogEndpoint::debug(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	log.log(LOG_LEVEL_DEBUG, process, fmt, args);
	va_end(args);
}

void LogEndpoint::info(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	log.log(LOG_LEVEL_INFO, process, fmt, args);
	va_end(args);
}

void LogEndpoint::warning(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	log.log(LOG_LEVEL_WARN, process, fmt, args);
	va_end(args);
}

void LogEndpoint::error(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	log.log(LOG_LEVEL_ERROR, process, fmt, args);
	va_end(args);
}

void LogEndpoint::critical(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	log.log(LOG_LEVEL_CRIT, process, fmt, args);
	va_end(args);
}
