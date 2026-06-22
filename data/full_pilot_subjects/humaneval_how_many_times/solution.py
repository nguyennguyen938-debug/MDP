def solve(string: str, substring: str) -> int:
    return sum(1 for i in range(len(string)) if string.startswith(substring, i))
