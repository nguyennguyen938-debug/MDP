int solve(const int *xs, int n) { int total = 0; for (int i = 0; i < n; i++) total += xs[i] < 0 ? -xs[i] : xs[i]; return total; }
