def solve(items: list[int]) -> int | None:
    unique = sorted(set(items))
    return unique[-2] if len(unique) >= 2 else None
