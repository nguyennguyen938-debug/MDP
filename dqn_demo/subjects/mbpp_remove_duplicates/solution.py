def solve(items: list[int]) -> list[int]:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out
