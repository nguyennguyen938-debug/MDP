def solve(text: str) -> list[str]:
    return [text[:i] for i in range(len(text) + 1)]
