/*
 * Copyright 2023-2025 komozoi
 * Original Creation Date: 2025-07-18
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


#include <gtest/gtest.h>
#include "fcntl.h"
#include "bigint.h"
#include "universaltime.h"


TEST(UnsignedFixedWidthBigIntTest, AddSpeed) {
	uint256_t a(0x55);
	a <<= 192;
	srand(42);
	for (int i = 0; i < 5000000; i++) {
		uint256_t b(0);
		b.data.rawBytes[rand() & 31] = rand();
		a += b;
		a += 67;
		a += b;
	}

	EXPECT_EQ(a, uint256_t("0x45A0285D15F95D6Db6F8D7b69934232b8bA67bc43ce40c70A3A6F3318D88326c"));
}

TEST(UnsignedFixedWidthBigIntTest, MulSpeed) {
	uint256_t a(0x55);
	a <<= 192;
	srand(42);
	for (int i = 0; i < 1000000; i++) {
		uint256_t b(0);
		b.data.rawBytes[rand() & 31] = rand();
		b.data.rawBytes[rand() & 31] = rand();
		b = b * b;
		a = (a + 17) * (b + 13);
	}
	for (int i = 0; i < 500000; i++)
		a = a * (a + 17);

	EXPECT_EQ(a, uint256_t("0xA6cb8c90AA2F76816928357e9453e02b6740DD86DA7684671774FD33c9327316"));
}


TEST(UnsignedFixedWidthBigIntTest, PowCorrectness) {
	uint256_t x1("0x1234567890AB");
	uint256_t x2("0xFFFFFFFFFFFF");
	uint256_t x3("0xABCDEF123456");

	EXPECT_EQ(x1.pow(2), uint256_t("0x14b66dc32880b72d610d239"));
	EXPECT_EQ(x1.pow(3), uint256_t("0x1790fc50e94bdb5b80e340f23424347c13"));
	EXPECT_EQ(x1.pow(4), uint256_t("0x1ad0326bee178e7e22c39f9f9d3de6c9c41563fc190b1"));

	EXPECT_EQ(x2.pow(2), uint256_t("0xfffffffffffe000000000001"));
	EXPECT_EQ(x2.pow(3), uint256_t("0xfffffffffffd000000000002ffffffffffff"));
	EXPECT_EQ(x2.pow(4), uint256_t("0xfffffffffffc000000000005fffffffffffc000000000001"));

	EXPECT_EQ(x3.pow(2), uint256_t("0x734cc30b1455b4bffacb0ce4"));
	EXPECT_EQ(x3.pow(3), uint256_t("0x4d61066d7383ea8931e1a45a895342dca498"));
	EXPECT_EQ(x3.pow(4), uint256_t("0x33ee0e405772f4bd1fa6d7a4e8c14117ea371272c23e2b10"));
}


/*TEST(UnsignedFixedWidthBigIntTest, PowSpeed) {
	const int N = 1000000;

	uint64_t start = millis_since_epoch();

	uint256_t accumulator("0xABCDEF123456");
	for (int i = 0; i < N; ++i) {
		uint256_t r = accumulator.pow(3) >> 84;
		accumulator = accumulator ^ r;
	}

	uint64_t end = millis_since_epoch();
	printf("Computed cubes of %d values in %llu ms\n", N, end - start);
}*/

TEST(UnsignedFixedWidthBigIntTest, RootCorrectness) {
	uint256_t x1("0x0123456789ABCDEF01234567");
	uint256_t x2("0xFFFFFFFF00000000ABCDEF01");
	uint256_t x3("0x783924abc37678847777fcba");

	uint256_t x5("0x1");
	EXPECT_EQ(x5.root(3), uint256_t("0x1"));

	uint256_t x4("0x3d731c3ff4324e0");
	EXPECT_EQ(x4.root(3), uint256_t("0x9f193"));

	EXPECT_EQ(x1.root(2), uint256_t("0x111111111111"));
	EXPECT_EQ(x1.root(3), uint256_t("0x2a170b82"));
	EXPECT_EQ(x1.root(4), uint256_t("0x421952"));

	EXPECT_EQ(x2.root(2), uint256_t("0xffffffff8000"));
	EXPECT_EQ(x2.root(3), uint256_t("0xffffffff"));
	EXPECT_EQ(x2.root(4), uint256_t("0xffffff"));

	EXPECT_EQ(x3.root(2), uint256_t("0xaf6f24e0603c"));
	EXPECT_EQ(x3.root(3), uint256_t("0xc6fc718e"));
	EXPECT_EQ(x3.root(4), uint256_t("0xd3ec28"));
}


/*TEST(UnsignedFixedWidthBigIntTest, RootSpeed) {
	const int N = 100000;

	uint64_t start = millis_since_epoch();

	uint256_t accumulator("0x1234567890ABCDEF1234567890ABCDEF");
	for (int i = 0; i < N; ++i) {
		uint256_t r = accumulator.root(3) << 82;
		accumulator = accumulator ^ r;
	}

	EXPECT_EQ(accumulator, uint256_t("0x1222312d4f0bcdef1234567890abcdef"));

	uint64_t end = millis_since_epoch();
	printf("Computed cube roots of %d values in %llu ms\n", N, end - start);
}*/


TEST(UnsignedFixedWidthBigIntTest, Shifts) {
	uint256_t a("0xFFD5A3FDF939e83D35144eeD319A891609DD4AA72e13e4742680DA9b237Ac613");
	EXPECT_EQ(a >> 91, uint256_t("0x1ffab47fbf273d07a6a289dda6335122c13ba954e5"));
	EXPECT_EQ(a.sar(91), uint256_t("0xfffffffffffffffffffffffffab47fbf273d07a6a289dda6335122c13ba954e5"));
	EXPECT_EQ(a << 91, uint256_t("0x698cd448b04eea5539709f23a13406d4d91bd630980000000000000000000000"));
}

TEST(UnsignedFixedWidthBigIntTest, Divide) {
	uint256_t numerator = "0x6935282358963433459348abcdef1ee7";
	uint256_t denominator = "0x62347874";
	uint256_t resultExpected = "0x011241296e03611cd7257cbccb";
	uint256_t result = numerator / denominator;
	EXPECT_EQ(result, resultExpected);

	numerator = "0x70A0823100000000000000000000000080A64C6d7F12C47b7C66C5b4e20e72bC";
	denominator = "0x0100000000000000000000000000000000000000000000000000000000";
	resultExpected = "0x70A08231";
	result = numerator / denominator;
	EXPECT_EQ(result, resultExpected);
}

TEST(UnsignedFixedWidthBigIntTest, Modulo) {
	uint256_t numerator("0x65aef76506eaf532fea24877eaf685785efa");
	uint256_t denominator("0x0cb8371865bc734976");
	uint256_t resultExpected("0x0973366e47deee1d4a");
	uint256_t result = numerator % denominator;
	EXPECT_EQ(result, resultExpected);
}


TEST(UnsignedFixedWidthBigIntTest, Multiply) {
	uint256_t a(0x41);
	uint256_t b("0xFFD5A3FDF939e83D35144eeD319A891609DD4AA72e13e4742680DA9b237Ac613");
	uint256_t resultExpected("0xf53ea37c47b3f78a7a280a39983cce98812ff472b30d017dc6b78164022c4ad3");
	uint256_t result = a * b;
	EXPECT_EQ(result, resultExpected);
}

TEST(UnsignedFixedWidthBigIntTest, SignedLessThan_PositiveVsNegative) {
	uint256_t a(1); // +1
	uint256_t b = -a; // -1

	EXPECT_TRUE(b.signedLessThan(a));
	EXPECT_FALSE(a.signedLessThan(b));
}

TEST(UnsignedFixedWidthBigIntTest, SignedLessThan_NegativeVsNegative) {
	uint256_t a = -uint256_t(5); // -5
	uint256_t b = -uint256_t(3); // -3

	EXPECT_TRUE(a.signedLessThan(b)); // -5 < -3
	EXPECT_FALSE(b.signedLessThan(a));
}

TEST(UnsignedFixedWidthBigIntTest, SignedDiv_PositiveByPositive) {
	uint256_t a(10);
	uint256_t b(2);

	auto result = a.sdiv(b);
	EXPECT_EQ(result, uint256_t(5));
}

TEST(UnsignedFixedWidthBigIntTest, SignedDiv_NegativeByPositive) {
	uint256_t a = -uint256_t(10);
	uint256_t b(2);

	auto result = a.sdiv(b);
	EXPECT_EQ(result, -uint256_t(5));
}

TEST(UnsignedFixedWidthBigIntTest, SignedDiv_PositiveByNegative) {
	uint256_t a(10);
	uint256_t b = -uint256_t(2);

	auto result = a.sdiv(b);
	EXPECT_EQ(result, -uint256_t(5));
}

TEST(UnsignedFixedWidthBigIntTest, SignedDiv_NegativeByNegative) {
	uint256_t a = -uint256_t(10);
	uint256_t b = -uint256_t(2);

	auto result = a.sdiv(b);
	EXPECT_EQ(result, uint256_t(5));
}

TEST(UnsignedFixedWidthBigIntTest, SignedMod_PositiveByPositive) {
	uint256_t a(10);
	uint256_t b(3);

	auto result = a.smod(b);
	EXPECT_EQ(result, uint256_t(1));
}

TEST(UnsignedFixedWidthBigIntTest, SignedMod_NegativeByPositive) {
	uint256_t a = -uint256_t(10);
	uint256_t b(3);

	auto result = a.smod(b);
	EXPECT_EQ(result, -uint256_t(1));
}

TEST(UnsignedFixedWidthBigIntTest, SignedMod_PositiveByNegative) {
	uint256_t a(10);
	uint256_t b = -uint256_t(3);

	auto result = a.smod(b);
	EXPECT_EQ(result, uint256_t(1));
}

TEST(UnsignedFixedWidthBigIntTest, SignedMod_NegativeByNegative) {
	uint256_t a = -uint256_t(10);
	uint256_t b = -uint256_t(3);

	auto result = a.smod(b);
	EXPECT_EQ(result, -uint256_t(1));
}


TEST(UnsignedFixedWidthBigIntTest, ToDouble) {
	// Note that these numbers purposely do not check for rounding, and that is intended.
	// We have no need for rounding.
	uint256_t a(0x41);
	uint256_t b("0xFFD5A3FDF939e83D35144eeD319A891609DD4AA72e13e4742680DA9b237Ac613");
	uint256_t c("0x7487298347897634897239467534875349347581");
	EXPECT_EQ(a.toDouble(), (double)0x41);
	EXPECT_EQ(b.toDouble(), 115717246645298340423446957891742553467495515860859582991948248975233360330752.0);
	EXPECT_EQ(c.toDouble(), 665257146293169136475648336830346788477288609153.0);
	EXPECT_EQ(uint256_t(0).toDouble(), 0.0);
}


TEST(UnsignedFixedWidthBigIntTest, FromDouble) {
	// Note that these numbers purposely do not check for rounding, and that is intended.
	// We have no need for rounding.
	uint256_t a(0x41);
	uint256_t b("0xFFD5A3FDF939e800000000000000000000000000000000000000000000000000");
	uint256_t c("0x7487298347897400000000000000000000000000");
	EXPECT_EQ(a, uint256_t(65.0));
	EXPECT_EQ(b, uint256_t(115717246645298340423446957891742553467495515860859582991948248975233360330752.0));
	EXPECT_EQ(c, uint256_t(665257146293169136475648336830346788477288609153.0));
	EXPECT_EQ(uint256_t(0), uint256_t(0.0));
}


TEST(UnsignedFixedWidthBigIntTest, MulDouble) {
	// Note that these numbers purposely do not check for rounding, and that is intended.
	// We have no need for rounding.
	uint256_t a(0x41);
	uint256_t b("0xFFD5A3FDF939e83D35144eeD319A891609DD4AA72e13e4742680DA9b237Ac613");
	uint256_t c("0x192168001001");
	EXPECT_EQ(a * 0.5, uint256_t(0x41 / 2));
	EXPECT_EQ(a * 0.25, uint256_t(0x41 / 4));
	EXPECT_EQ(a * 8713052688097476608.0, uint256_t("0x1eb3a9cba563500000"));
	EXPECT_EQ(a * 0.015384615384615386, uint256_t(1));
	EXPECT_EQ(b * 0.00000000000012891, uint256_t("0x2442f0c5236768da9a93db2e9ea36fef06f8e4a930a5cea03ed524"));
	EXPECT_EQ(c * 807.621, uint256_t("0x4f47e9fd63a400"));
	EXPECT_EQ(c * 8713052688e31, uint256_t("0x1922c0DD776453224c84F40c5000000000000000000000"));
	EXPECT_EQ(c * 0.0, uint256_t(0));
	EXPECT_EQ(c * 1e98, uint256_t(0));
}

