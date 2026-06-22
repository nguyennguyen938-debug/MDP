int solve(int x) { int y = x ^ 7; return (y & 7) ? y : -y; }
