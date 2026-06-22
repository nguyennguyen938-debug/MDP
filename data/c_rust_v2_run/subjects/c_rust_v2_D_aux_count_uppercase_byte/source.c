int solve(const char *s) { int c = 0; for (int i = 0; s[i]; i++) { unsigned char b = (unsigned char)s[i]; if (b >= 'A' && b <= 'Z') c++; } return c; }
