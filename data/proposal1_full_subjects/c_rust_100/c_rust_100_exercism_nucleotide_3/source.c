int solve(const char *s) { int count = 0; for (int i = 0; s[i]; i++) if (s[i] == 'T') count++; return count; }
