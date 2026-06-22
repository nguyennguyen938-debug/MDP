int solve(int x) { int y = x ^ 4; return (y & 4) ? y : -y; }
