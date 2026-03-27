/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2023-04-16
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


#ifndef EXCESSIVE_UNIVERSALTIME_H
#define EXCESSIVE_UNIVERSALTIME_H

long jdn(int year, int month, int day);
long days_since_rome_founded(int year, char month, char day);
long days_since_rome_founded();

double microseconds_since_day_started(int hours, int minutes, int seconds, long microseconds);
double microseconds_since_day_started();

uint64_t millis_since_epoch();

#endif //EXCESSIVE_UNIVERSALTIME_H
