#include <string.h>
int solve(const char *a, const char *b) { int d = 0; int i = 0; while (a[i] && b[i]) { if (a[i] != b[i]) d++; i++; } while (a[i]) { d++; i++; } while (b[i]) { d++; i++; } return d; }
