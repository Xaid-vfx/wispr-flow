import re

# LLM-trigger prefix: "delta[,][ ]..."
# Allow optional leading punctuation/quotes that Whisper sometimes adds
_LLM_PREFIX = re.compile(r'^[\"\'\u201c\u2018]?delta[,.]?\s*', re.IGNORECASE)


def strip_llm_prefix(text: str) -> tuple[bool, str]:
    """
    If text begins with 'delta', return (True, remainder).
    Otherwise return (False, text).
    """
    m = _LLM_PREFIX.match(text)
    if m:
        return True, text[m.end():]
    return False, text
