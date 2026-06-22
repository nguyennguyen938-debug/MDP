def solve(numbers: list[int]) -> list[int]:
    total = 0
    product = 1
    for number in numbers:
        total += number
        product *= number
    return [total, product]
