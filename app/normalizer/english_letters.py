"""
English Letter Pronunciation Dictionary for TTS.

Maps each English letter to its Vietnamese phonetic approximation
of the English pronunciation (not Vietnamese alphabet pronunciation).

Used to spell out English abbreviations letter-by-letter:
    TTS → "ti ti ét"
    AI  → "ây ai"
    KPI → "kây pi ai"
    CEO → "xi yi âu"
"""

# Vietnamese phonetic approximation of English letter pronunciation
# User can customize this dictionary as needed
ENGLISH_LETTER_PRONUNCIATION = {
    "A": "ây", "B": "bi", "C": "xi", "D": "đi", "E": "yi",
    "F": "ép", "G": "ji", "H": "hếch", "I": "ai", "J": "jây",
    "K": "kây", "L": "eo", "M": "em", "N": "en", "O": "âu",
    "P": "pi", "Q": "kiu", "R": "ah", "S": "ét", "T": "ti",
    "U": "iu", "V": "vi", "W": "đắp liu", "X": "éch", "Y": "oai",
    "Z": "zi",
}


def spell_out_abbreviation(abbr: str) -> str:
    """
    Convert an abbreviation to its letter-by-letter Vietnamese pronunciation.

    Each letter is looked up in ENGLISH_LETTER_PRONUNCIATION and joined with spaces.
    Non-letter characters are skipped.

    Args:
        abbr: The abbreviation to spell out (e.g., "TTS", "AI", "KPI")

    Returns:
        Space-separated pronunciation string (e.g., "ti ti ét", "ây ai")

    Examples:
        >>> spell_out_abbreviation("TTS")
        'ti ti ét'
        >>> spell_out_abbreviation("AI")
        'ây ai'
        >>> spell_out_abbreviation("KPI")
        'kây pi ai'
        >>> spell_out_abbreviation("CEO")
        'xi i âu'
    """
    parts = []
    for ch in abbr:
        upper_ch = ch.upper()
        if upper_ch in ENGLISH_LETTER_PRONUNCIATION:
            parts.append(ENGLISH_LETTER_PRONUNCIATION[upper_ch])
    return " ".join(parts)
