def solve(matrix: list[list[int]]) -> int:
    return sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0]) if matrix else 0)))
