#include <stdbool.h>
int powi(int b, int e) { int out = 1; for (int i = 0; i < e; i++) out *= b; return out; }
bool solve(int n) { if (n < 0) return false; int original = n; int tmp = n; int digits = 0; do { digits++; tmp /= 10; } while (tmp); tmp = n; int sum = 0; do { sum += powi(tmp % 10, digits); tmp /= 10; } while (tmp); return sum == original; }
