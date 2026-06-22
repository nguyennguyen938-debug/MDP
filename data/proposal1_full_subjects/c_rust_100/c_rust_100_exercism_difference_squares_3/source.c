int solve(int n) { int sum = 0; int squares = 0; for (int i = 1; i <= n; i++) { sum += i; squares += i * i; } return sum * sum - squares; }
