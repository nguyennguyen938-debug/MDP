def solve(text: str) -> dict[str, int]:
    return {word: len(word) for word in text.split()}
