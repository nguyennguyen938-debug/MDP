int solve(int x) { int y = x ^ 8; return (y & 8) ? y : -y; }
