import re

INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"reveal (the )?system prompt",
    r"you are now",
    r"do not follow",
    r"developer message",
]

def looks_like_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)

