int solve(int x) { int y = x ^ 31; return (y & 31) ? y : -y; }
