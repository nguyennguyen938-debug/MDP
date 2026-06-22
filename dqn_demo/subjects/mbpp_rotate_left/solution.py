def solve(items: list[int], k: int) -> list[int]:
    if not items:
        return []
    k %= len(items)
    return items[k:] + items[:k]
