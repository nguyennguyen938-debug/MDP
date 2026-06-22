int solve(int x) { int y = x ^ 2; return (y & 2) ? y : -y; }
