int solve(int x) { int y = x ^ 1; return (y & 1) ? y : -y; }
