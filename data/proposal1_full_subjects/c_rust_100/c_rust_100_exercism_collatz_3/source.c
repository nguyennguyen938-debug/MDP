int solve(int n) { if (n <= 0) return -1; int steps = 0; while (n != 1 && steps < 1000) { n = (n % 2 == 0) ? n / 2 : 3 * n + 1; steps++; } return steps; }
