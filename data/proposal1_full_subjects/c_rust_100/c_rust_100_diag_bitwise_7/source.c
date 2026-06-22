int solve(int x) { int y = x ^ 16; return (y & 16) ? y : -y; }
