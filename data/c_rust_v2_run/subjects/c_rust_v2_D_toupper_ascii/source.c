int solve(const char *s) { int c = 0; for (int i = 0; s[i]; i++) if (s[i] >= 'A' && s[i] <= 'Z') c++; return c; }
