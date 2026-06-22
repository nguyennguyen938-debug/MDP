int solve(const char *s) { int n = 0; while (s[n]) n++; int sum = 0; for (int i = n - 1; i >= 0; i--) sum += (unsigned char)s[i]; return sum; }
