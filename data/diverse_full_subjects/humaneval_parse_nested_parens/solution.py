def solve(paren_string: str) -> list[int]:
    out = []
    depth = 0
    best = 0
    for ch in paren_string:
        if ch == '(':
            depth += 1
            best = max(best, depth)
        elif ch == ')':
            depth -= 1
            if depth == 0:
                out.append(best)
                best = 0
    return out
