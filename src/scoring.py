"""Character error rate (CER), shared by the demo and scripts/score.py so they always agree.

Punctuation and spaces are ignored: only letters, digits and Chinese characters are compared.
"""


def normalize(text: str) -> str:
    """Keep only letters/digits/CJK characters, so punctuation doesn't count."""
    return "".join(ch for ch in text if ch.isalnum())


def align(reference: str, prediction: str) -> list[tuple[str | None, str | None]]:
    """Character-level alignment (Levenshtein). Returns (ref_char, pred_char) pairs;
    None marks a missing character (deletion) or an extra one (insertion)."""
    ref, hyp = normalize(reference), normalize(prediction)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((ref[i - 1], None)); i -= 1
        else:
            pairs.append((None, hyp[j - 1])); j -= 1
    return pairs[::-1]


def score(reference: str, prediction: str) -> tuple[float, int, int]:
    """(CER in percent, characters correct, reference length)."""
    pairs = align(reference, prediction)
    errors = sum(r != h for r, h in pairs)
    correct = sum(r == h for r, h in pairs)
    length = max(len(normalize(reference)), 1)
    return 100 * errors / length, correct, length
