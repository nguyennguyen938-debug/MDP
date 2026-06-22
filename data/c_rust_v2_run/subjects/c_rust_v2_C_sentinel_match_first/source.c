int solve(const int *xs, int n, int needle) { for (int i = 0; i < n; i++) if (xs[i] == needle) return i; return -1; }
