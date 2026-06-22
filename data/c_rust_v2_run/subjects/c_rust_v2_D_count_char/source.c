int solve(const char *s, int byte) { int c = 0; for (int i = 0; s[i]; i++) if ((unsigned char)s[i] == (unsigned char)byte) c++; return c; }
