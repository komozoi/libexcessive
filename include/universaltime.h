//
// Created by nuclaer on 4/16/23.
//

#ifndef EXCESSIVE_UNIVERSALTIME_H
#define EXCESSIVE_UNIVERSALTIME_H

long jdn(int year, int month, int day);
long days_since_rome_founded(int year, char month, char day);
long days_since_rome_founded();

double microseconds_since_day_started(int hours, int minutes, int seconds, long microseconds);
double microseconds_since_day_started();

uint64_t millis_since_epoch();

#endif //EXCESSIVE_UNIVERSALTIME_H
