"""
Special Token Detector for Hybrid Normalizer.

Scans text that has already been through rule-based normalization and
identifies tokens that should be sent to the LLM for context-aware
normalization. These include:
  - Uppercase abbreviations (AI, KPI, CEO, FCC, ...)
  - Special/math symbols (², √, ±, ≥, Δ, ∑, ...)
  - Mixed alphanumeric codes (802.11ax, 5G, Wi-Fi 6E, ...)
  - Residual single special characters (@, #, $, &, ...)
  - Short foreign-language words that survived rule-based processing
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SpecialToken:
    """A token detected as needing LLM normalization."""
    text: str       # The matched text
    start: int      # Start position in source string
    end: int        # End position in source string


# ── Regex patterns for special tokens ──────────────────────────────────

# 1. Uppercase abbreviations: 2+ consecutive uppercase ASCII letters
#    (but NOT Vietnamese uppercase words — they typically have diacritics or are lowercase)
#    Matches: AI, KPI, CEO, FCC, RoHS, TP.HCM, IEEE
_ABBREV_RE = re.compile(
    r'\b[A-Z][A-Za-z]*\.[A-Z][A-Za-z]*(?:\.[A-Z][A-Za-z]*)*\b'  # TP.HCM, U.S.A
    r'|'
    r'\b[A-Z]{2,}(?:[a-z]+[A-Z]+[a-z]*)*\b'  # AI, KPI, RoHS, IoT
)

# 2. Mathematical / special symbols (unicode characters that rule-based usually can't handle)
_MATH_SYMBOL_RE = re.compile(
    r'[²³⁴⁵⁶⁷⁸⁹⁰¹'
    r'√∛∜±∓×÷≠≈≡≤≥≪≫∞∝∂∇∆Δ∑∏∫∬∮'
    r'αβγδεζηθικλμνξπρσςτυφχψω'
    r'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ'
    r'ℵℝℤℚℕℂ∈∉⊂⊃⊆⊇∪∩∅∧∨¬⊕⊗'
    r'→←↔⇒⇐⇔↑↓'
    r'°′″℃℉'
    r']+'
)

# 3. Mixed alphanumeric codes: letters+digits interleaved
#    Matches: 802.11ax, A1B2, Wi-Fi 6E, 5G, H.264
_ALPHANUMERIC_CODE_RE = re.compile(
    r'\b(?:[A-Za-z]+[-.]?\d+[A-Za-z0-9.]*'  # starts with letters: A1B2, H.264
    r'|\d+[.]\d+[A-Za-z]+)'                  # starts with digits: 802.11ax
    r'\b'
)

# 4. Residual special characters that rule-based should have removed but didn't
_RESIDUAL_SPECIAL_RE = re.compile(
    r'[#\$&\*@~\^\\|{}[\]<>]'
)

# 5. Math expressions: sequences containing math operators between terms
#    e.g., "ax² + bx + c = 0", "Δ = b²−4ac ≥ 0"
_MATH_EXPR_RE = re.compile(
    r'(?:[a-zA-Zα-ωΑ-Ω0-9²³⁴⁰¹⁵⁶⁷⁸⁹√Δ∑∏()]+\s*'
    r'[+\-−×÷=≠≈≡≤≥<>±∓/]\s*'
    r'[a-zA-Zα-ωΑ-Ω0-9²³⁴⁰¹⁵⁶⁷⁸⁹√Δ∑∏()]+)'
    r'(?:\s*[+\-−×÷=≠≈≡≤≥<>±∓/]\s*'
    r'[a-zA-Zα-ωΑ-Ω0-9²³⁴⁰¹⁵⁶⁷⁸⁹√Δ∑∏()]+)*'
)


def _merge_overlapping(tokens: List[SpecialToken]) -> List[SpecialToken]:
    """Merge overlapping or adjacent special tokens into single spans."""
    if not tokens:
        return []
    # Sort by start position
    sorted_tokens = sorted(tokens, key=lambda t: (t.start, -t.end))
    merged = [sorted_tokens[0]]
    for tok in sorted_tokens[1:]:
        prev = merged[-1]
        if tok.start <= prev.end:
            # Overlapping or adjacent — extend
            if tok.end > prev.end:
                merged[-1] = SpecialToken(
                    text=prev.text[:tok.start - prev.start] + tok.text + prev.text[tok.end - prev.start:] if tok.end <= prev.end else prev.text + tok.text[prev.end - tok.start:],
                    start=prev.start,
                    end=max(prev.end, tok.end),
                )
                # Re-read from source would be better — but we fix text in caller
        else:
            merged.append(tok)
    return merged


def detect_special_tokens(text: str) -> List[SpecialToken]:
    """
    Detect special tokens in text that need LLM normalization.

    Args:
        text: Text that has already been through rule-based normalization.

    Returns:
        List of SpecialToken objects, sorted by position, non-overlapping.
    """
    found: List[SpecialToken] = []

    # Pattern priority: math expressions first (they subsume individual symbols)
    for m in _MATH_EXPR_RE.finditer(text):
        matched = m.group().strip()
        # Only include if it actually contains special math symbols or operators
        if re.search(r'[²³⁴⁰¹⁵⁶⁷⁸⁹√±∓×÷≠≈≡≤≥Δ∑∏=]', matched):
            found.append(SpecialToken(text=matched, start=m.start(), end=m.end()))

    # Individual math/special symbols (not already inside a math expression)
    for m in _MATH_SYMBOL_RE.finditer(text):
        found.append(SpecialToken(text=m.group(), start=m.start(), end=m.end()))

    # Uppercase abbreviations
    for m in _ABBREV_RE.finditer(text):
        abbrev = m.group()
        # Skip if it's a common Vietnamese word written in caps (unlikely after normalize)
        # Skip single-char matches
        if len(abbrev) >= 2:
            found.append(SpecialToken(text=abbrev, start=m.start(), end=m.end()))

    # Mixed alphanumeric codes
    for m in _ALPHANUMERIC_CODE_RE.finditer(text):
        found.append(SpecialToken(text=m.group(), start=m.start(), end=m.end()))

    # Residual special characters
    for m in _RESIDUAL_SPECIAL_RE.finditer(text):
        found.append(SpecialToken(text=m.group(), start=m.start(), end=m.end()))

    # Merge overlapping spans and sort
    merged = _merge_overlapping(found)
    # Re-read actual text from source to fix text field after merge
    for tok in merged:
        tok.text = text[tok.start:tok.end]

    return merged
