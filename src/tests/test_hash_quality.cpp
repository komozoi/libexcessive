/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2026-4-6
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

#include "hash.h"
#include "ds/ArrayList.h"
#include <gtest/gtest.h>
#include <chrono>
#include <string>
#include <random>
#include <bitset>
#include <cstdio>

#include "ds/HashSet.h"


#if defined(__SSE2__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__ARM_NEON)

TEST(HashQualityTest, SpeedTest) {
    const size_t sizeBytes = 5 * 1024 * 1024; // 5 MiB
    const size_t sizeWords = sizeBytes / 8;
    ArrayList<uint64_t> data((int)sizeWords);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < sizeWords; ++i) {
        data.add(rng());
    }

    auto start = std::chrono::high_resolution_clock::now();
    size_t hash = excessiveFastHash(data.getMemory(), sizeWords);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double seconds = duration.count() / 1e9;
    double gib = static_cast<double>(sizeBytes) / (1024 * 1024 * 1024);
    double gibs = seconds > 0 ? gib / seconds : 0;
    
    printf("[          ] Hash processing 5MiB took %.2f ms (%.2f GiB/s)\n", seconds * 1000.0, gibs);
    
    EXPECT_LT(seconds * 1000.0, 12.0) << "Hash processing 5MiB took too long";
    // Use the hash to prevent optimization
    ASSERT_NE(hash, 0);
}

#endif

TEST(HashQualityTest, ThreeLetterCollisions) {
    HashSet<size_t> hashes(52 * 52 * 52 + 500);
    int collisions = 0;
    std::string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    for (char c1 : letters) {
        for (char c2 : letters) {
            for (char c3 : letters) {
                char s[8] = {c1, c2, c3, 0, 0, 0, 0, 0}; // Pad to 8 bytes for 1 word
                size_t h = excessiveFastHash(s, 1);
                if (hashes.add(h)) {
                    collisions++;
                }
            }
        }
    }

    printf("[          ] Found %d collisions in %u 3-letter strings\n", collisions, (unsigned int)(52 * 52 * 52));
    EXPECT_LT(collisions, 15) << "Found too many collisions in 3-letter strings";
}

TEST(HashQualityTest, BitDiffTest) {
    const int numPairs = 100000;
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<uint64_t> distWord(0, (uint64_t)-1);
    std::uniform_int_distribution<size_t> distLen(1, 10); // in words

    long totalDiffBits = 0;

    for (int i = 0; i < numPairs; ++i) {
        size_t len = distLen(rng);
        ArrayList<uint64_t> s1((int)len);
        for (size_t j = 0; j < len; ++j) s1.add(distWord(rng));

        ArrayList<uint64_t> s2 = s1;
        std::uniform_int_distribution<int> action(0, 1); // 0: mutate, 1: add word
        if (action(rng) == 0) {
            // mutate
            std::uniform_int_distribution<size_t> posDist(0, len - 1);
            size_t pos = posDist(rng);
            uint64_t old = s2.get((int)pos);
            while (s2.get((int)pos) == old) s2.set((int)pos, distWord(rng));
        } else {
            // add
            s2.add(distWord(rng));
        }

        size_t h1 = excessiveFastHash(s1.getMemory(), s1.size());
        size_t h2 = excessiveFastHash(s2.getMemory(), s2.size());

        size_t diff = h1 ^ h2;
        totalDiffBits += std::bitset<64>(diff).count();
    }

    double avgDiffBits = static_cast<double>(totalDiffBits) / numPairs;
    printf("[          ] Average bit difference (mutation/addition): %.2f\n", avgDiffBits);
    EXPECT_GT(avgDiffBits, 24) << "Average bit difference too low";
}

TEST(HashQualityTest, SingleBitMutationDiffTest) {
    const int numPairs = 100000;
    std::mt19937_64 rng(54321);
    std::uniform_int_distribution<uint64_t> distWord(0, (uint64_t)-1);
    std::uniform_int_distribution<size_t> distLen(1, 10); // in words

    long totalDiffBits = 0;

    for (int i = 0; i < numPairs; ++i) {
        size_t len = distLen(rng);
        ArrayList<uint64_t> s1((int)len);
        for (size_t j = 0; j < len; ++j) s1.add(distWord(rng));

        ArrayList<uint64_t> s2 = s1;
        
        // mutate a single random bit
        std::uniform_int_distribution<size_t> posDist(0, len - 1);
        std::uniform_int_distribution<int> bitDist(0, 63);
        
        size_t pos = posDist(rng);
        int bit = bitDist(rng);
        
        s2.set((int)pos, s2.get((int)pos) ^ (1ULL << bit));

        size_t h1 = excessiveFastHash(s1.getMemory(), s1.size());
        size_t h2 = excessiveFastHash(s2.getMemory(), s2.size());

        size_t diff = h1 ^ h2;
        totalDiffBits += std::bitset<64>(diff).count();
    }

    double avgDiffBits = static_cast<double>(totalDiffBits) / numPairs;
    printf("[          ] Average bit difference (single bit mutation): %.2f\n", avgDiffBits);
    EXPECT_GT(avgDiffBits, 24) << "Average bit difference too low for single bit mutation";
}
