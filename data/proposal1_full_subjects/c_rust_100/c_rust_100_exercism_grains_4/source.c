int solve(int square) { if (square <= 0 || square > 30) return 0; return 1 << (square - 1); }
