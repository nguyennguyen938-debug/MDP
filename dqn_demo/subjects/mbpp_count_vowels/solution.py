def solve(text: str) -> int:
    return sum(1 for ch in text.lower() if ch in 'aeiou')
