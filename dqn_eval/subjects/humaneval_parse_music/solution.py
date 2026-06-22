def solve(music_string: str) -> list[int]:
    mapping = {'o': 4, 'o|': 2, '.|': 1}
    return [mapping[token] for token in music_string.split()]
