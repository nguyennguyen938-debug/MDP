#include <string.h>
int solve(const char *color) { if (strcmp(color, "black") == 0) return 0; if (strcmp(color, "brown") == 0) return 1; if (strcmp(color, "red") == 0) return 2; if (strcmp(color, "green") == 0) return 5; if (strcmp(color, "blue") == 0) return 6; if (strcmp(color, "white") == 0) return 9; return -1; }
