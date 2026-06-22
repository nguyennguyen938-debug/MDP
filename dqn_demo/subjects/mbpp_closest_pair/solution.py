def solve(items: list[int], target: int) -> list[int] | None:
    best = None
    best_dist = None
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            dist = abs(items[i] + items[j] - target)
            if best_dist is None or dist < best_dist:
                best = [items[i], items[j]]
                best_dist = dist
    return best
