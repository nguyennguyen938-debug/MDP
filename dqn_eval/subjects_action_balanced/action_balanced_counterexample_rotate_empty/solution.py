def solve(items: list[int], k: int) -> list[int]:
    k %= len(items)
    return items[k:] + items[:k]
