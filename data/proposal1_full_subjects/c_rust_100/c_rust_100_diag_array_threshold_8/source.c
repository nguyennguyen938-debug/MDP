int solve(const int *xs, int n) { int count = 0; for (int i = 0; i < n; i++) if (xs[i] >= 3) count++; return count; }
