int solve(const char *s) { int c = 0; for (int i = 0; s[i]; i++) if ((unsigned char)s[i] >= 128) c++; return c; }
