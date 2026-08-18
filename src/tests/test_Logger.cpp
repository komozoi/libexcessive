/*
 * Copyright 2023-2026 komozoi
 * Original Creation Date: 2026-08-18
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


#include "Logger.h"
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <string>
#include <stdexcept>


static mode_t currentUmask() {
	mode_t mask = umask(0);
	umask(mask);
	return mask;
}

static std::string todayLogName() {
	time_t now = time(0);
	struct tm* localTime = localtime(&now);
	char name[16];
	strftime(name, sizeof(name), "%Y-%m-%d.txt", localTime);
	return std::string(name);
}

static std::string readWholeFile(const std::string& path) {
	int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0)
		return "";
	char buf[4096];
	ssize_t n = read(fd, buf, sizeof(buf) - 1);
	close(fd);
	if (n < 0)
		return "";
	buf[n] = 0;
	return std::string(buf);
}

static void removeLogDir(const std::string& dir) {
	std::string logFile = dir + "/" + todayLogName();
	unlink(logFile.c_str());
	rmdir(dir.c_str());
}


class LoggerTest : public ::testing::Test {
protected:
	std::string root;

	void SetUp() override {
		char tmpl[] = "/tmp/libexcessive_logger_XXXXXX";
		char* created = mkdtemp(tmpl);
		ASSERT_NE(created, (char*)nullptr);
		root = created;
	}

	void TearDown() override {
		if (!root.empty())
			rmdir(root.c_str());
	}

	std::string child(const char* name) const {
		return root + "/" + name;
	}
};


TEST_F(LoggerTest, OpenFailureWhenPathIsFileThrows) {
	std::string path = child("notadir");
	int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
	ASSERT_GE(fd, 0);
	close(fd);

	EXPECT_THROW(Logger(path.c_str(), LOG_LEVEL_DEBUG, 5), std::runtime_error);

	unlink(path.c_str());
}

TEST_F(LoggerTest, OpenFailureWhenParentMissingThrows) {
	std::string path = child("missing") + "/nested";
	EXPECT_THROW(Logger(path.c_str(), LOG_LEVEL_DEBUG, 5), std::runtime_error);
}

TEST_F(LoggerTest, DefaultDirModeIsOwnerOnly) {
	std::string dir = child("default");
	{
		Logger logger(dir.c_str(), LOG_LEVEL_DEBUG, 5);
	}

	struct stat st;
	ASSERT_EQ(stat(dir.c_str(), &st), 0);
	EXPECT_EQ(st.st_mode & 0777, (mode_t)0700 & ~currentUmask());

	removeLogDir(dir);
}

TEST_F(LoggerTest, CustomDirModeIsApplied) {
	std::string dir = child("custom");
	{
		Logger logger(dir.c_str(), LOG_LEVEL_DEBUG, 5, 0750);
	}

	struct stat st;
	ASSERT_EQ(stat(dir.c_str(), &st), 0);
	EXPECT_EQ(st.st_mode & 0777, (mode_t)0750 & ~currentUmask());

	removeLogDir(dir);
}

TEST_F(LoggerTest, ExistingDirectoryPermsAreUnchanged) {
	std::string dir = child("existing");
	ASSERT_EQ(mkdir(dir.c_str(), 0755), 0);

	{
		Logger logger(dir.c_str(), LOG_LEVEL_DEBUG, 5, 0700);
	}

	struct stat st;
	ASSERT_EQ(stat(dir.c_str(), &st), 0);
	EXPECT_EQ(st.st_mode & 0777, (mode_t)0755 & ~currentUmask());

	removeLogDir(dir);
}

TEST_F(LoggerTest, WritesStartBannerAndMessage) {
	std::string dir = child("write");
	{
		Logger logger(dir.c_str(), LOG_LEVEL_DEBUG, 5);
		LogEndpoint ep(logger, "Test");
		ep.info("hello %d", 42);
	}

	std::string contents = readWholeFile(dir + "/" + todayLogName());
	EXPECT_NE(contents.find("Log started"), std::string::npos);
	EXPECT_NE(contents.find("hello 42"), std::string::npos);
	EXPECT_NE(contents.find("Log closed"), std::string::npos);

	struct stat st;
	ASSERT_EQ(stat((dir + "/" + todayLogName()).c_str(), &st), 0);
	EXPECT_EQ(st.st_mode & 0777, (mode_t)0600 & ~currentUmask());

	removeLogDir(dir);
}
