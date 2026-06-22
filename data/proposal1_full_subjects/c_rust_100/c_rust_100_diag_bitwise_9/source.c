int solve(int x) { int y = x ^ 63; return (y & 63) ? y : -y; }
