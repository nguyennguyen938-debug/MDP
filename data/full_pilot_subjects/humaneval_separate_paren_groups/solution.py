def solve(paren_string: str) -> list[str]:
    groups = []
    depth = 0
    current = []
    for ch in paren_string.replace(' ', ''):
        current.append(ch)
        depth += 1 if ch == '(' else -1
        if depth == 0 and current:
            groups.append(''.join(current))
            current = []
    return groups
