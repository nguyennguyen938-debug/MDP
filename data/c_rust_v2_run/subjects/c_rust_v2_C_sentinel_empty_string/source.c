int solve(const char *s) { if (s == 0 || s[0] == 0) return -1; int c = 0; for (int i = 0; s[i]; i++) c++; return c; }
