def solve(numbers: list[int], delimiter: int) -> list[int]:
    out = []
    for idx, number in enumerate(numbers):
        if idx:
            out.append(delimiter)
        out.append(number)
    return out
