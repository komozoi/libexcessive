//
// Created by nuclaer on 26.7.25.
//

#include "strutil.h"

#include "stdlib.h"


char* formatBinaryDataForHexdump(const uint8_t* data, int size, int columns) {
	if (data == nullptr)
		return strcpy((char*)malloc(7), "(null)");
	if (size == 0)
		return strcpy((char*)malloc(8), "(empty)");

	int nLines = 1;
	if (columns == -1 || columns >= size)
		columns = size;
	else
		nLines = (size + columns - 1) / columns;

	int outputLength = size*4 + (size-1)/8 + 3*nLines + 1;
	if (nLines > 1)
		outputLength += nLines * 7;
	char* output = (char*) malloc(outputLength);
	if (output == nullptr)
		return nullptr;

	char* p = output;
	const uint8_t* dp = data;

	const char* base = "0123456789abcdef";

	for (int line = 0; line < nLines; line++) {
		if (nLines > 1)
			p += sprintf(p, "%04x  ", dp - data);

		if (line + 1 == nLines)
			columns = size - columns * line;

		for (int i = 0; i < columns; i++) {
			if (i && (i & 7) == 0)
				*(p++) = ' ';
			*(p++) = base[dp[i] >> 4];
			*(p++) = base[dp[i] & 15];
			*(p++) = ' ';
		}

		*(p++) = ' ';
		*(p++) = '|';

		for (int i = 0; i < columns; i++) {
			uint8_t val = dp[i];
			char c = (val < 32 || val > 126) ? '.' : (char)val;
			*(p++) = c;
		}

		*(p++) = '|';

		if (nLines > 1)
			*(p++) = '\n';

		dp += columns;
	}

	*p = 0;

	return output;
}