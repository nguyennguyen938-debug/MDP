def solve(numbers: list[int]) -> list[int]:
    best = None
    out = []
    for number in numbers:
        best = number if best is None else max(best, number)
        out.append(best)
    return out
