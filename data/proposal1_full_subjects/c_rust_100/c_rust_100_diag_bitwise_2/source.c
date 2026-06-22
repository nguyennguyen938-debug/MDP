int solve(int x) { int y = x ^ 3; return (y & 3) ? y : -y; }
