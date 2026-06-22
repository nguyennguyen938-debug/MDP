int solve(const int *xs, int n) { int count = 0; for (int i = 0; i < n; i++) if (xs[i] >= 2) count++; return count; }
