//
// Created by komozoi on 4/16/23.
//

#include "stdint.h"
#include <time.h>
#include <sys/time.h>

// Function to calculate the Julian Day Number from a given date
long jdn(int year, int month, int day) {
	int a = (14 - month) / 12;
	int y = year + 4800 - a;
	int m = month + 12 * a - 3;
	long jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
	return jdn;
}

long days_since_rome_founded(int year, char month, char day) {
	long jdn_today = jdn(year, month, day);
	long jdn_rome_founded = jdn(-753 + 1, 4, 21); // Rome was founded on April 21, 753 BC, adding 1 to the year to account for no year 0 in Gregorian calendar
	return jdn_today - jdn_rome_founded;
}

// Function to compute the number of days since Rome was founded
long days_since_rome_founded() {
	time_t t = time(NULL);
	struct tm *today = localtime(&t);
	int year = today->tm_year + 1900;
	return days_since_rome_founded(year, today->tm_mon + 1, today->tm_mday);
}

double microseconds_since_day_started(int hours, int minutes, int seconds, long microseconds) {
	double total_microseconds = (double)hours * 3600.0 * 1e6
								+ (double)minutes * 60.0 * 1e6
								+ (double)seconds * 1e6
								+ (double)microseconds;

	return total_microseconds;
}

double microseconds_since_day_started() {
	struct timeval tv;
	gettimeofday(&tv, NULL);

	time_t rawtime = tv.tv_sec;
	struct tm *today = localtime(&rawtime);

	int hours = today->tm_hour;
	int minutes = today->tm_min;
	int seconds = today->tm_sec;

	return microseconds_since_day_started(hours, minutes, seconds, tv.tv_usec);
}

uint64_t millis_since_epoch() {
	struct timeval tv;
	gettimeofday(&tv, nullptr); // get current time
	uint64_t milliseconds_since_epoch =
			(uint64_t)(tv.tv_sec) * 1000 +  // convert seconds to milliseconds
			(uint64_t)(tv.tv_usec) / 1000;  // convert microseconds to milliseconds
	return milliseconds_since_epoch;
}
