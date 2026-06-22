def solve(items: list[int]) -> int:
    return sum(x for x in items if x % 2 == 0)
