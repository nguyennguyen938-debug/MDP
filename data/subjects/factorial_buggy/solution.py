def solve(n: int) -> int:
    out = 1
    for i in range(1, n):
        out *= i
    return out
