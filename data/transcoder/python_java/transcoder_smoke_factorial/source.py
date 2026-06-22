def factorial(n: int) -> int:
    if n <= 1:
        return 1
    acc = 1
    for k in range(2, n + 1):
        acc *= k
    return acc
