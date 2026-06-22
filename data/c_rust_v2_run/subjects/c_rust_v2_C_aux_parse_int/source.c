int solve(const char *s) { int n = 0; for (int i = 0; s[i]; i++) { if (s[i] < '0' || s[i] > '9') return -1; n = n*10 + (s[i]-'0'); } return n; }
