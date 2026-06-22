int solve(int x) { int y = x ^ 15; return (y & 15) ? y : -y; }
