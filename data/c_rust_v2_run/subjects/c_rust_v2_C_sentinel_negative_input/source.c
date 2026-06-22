int solve(int n) { if (n < 0) return -1; int s = 0; for (int i = 0; i <= n; i++) s += i; return s; }
