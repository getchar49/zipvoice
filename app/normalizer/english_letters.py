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

# Vietnamese pronunciation of digits (for alphanumeric abbreviations like VVS1, S2)
DIGIT_PRONUNCIATION = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}


def spell_out_abbreviation(abbr: str) -> str:
    """
    Convert an abbreviation to its letter-by-letter Vietnamese pronunciation.

    Each letter is looked up in ENGLISH_LETTER_PRONUNCIATION and joined with spaces.
    Digits are looked up in DIGIT_PRONUNCIATION.
    Multi-word abbreviations (e.g., "cron job") are processed word by word —
    each word is spelled out letter-by-letter, with words separated by spaces.

    Args:
        abbr: The abbreviation to spell out (e.g., "TTS", "AI", "VVS1", "S2")

    Returns:
        Space-separated pronunciation string (e.g., "ti ti ét", "ây ai",
        "vi vi ét một", "ét hai")

    Examples:
        >>> spell_out_abbreviation("TTS")
        'ti ti ét'
        >>> spell_out_abbreviation("AI")
        'ây ai'
        >>> spell_out_abbreviation("VVS1")
        'vi vi ét một'
        >>> spell_out_abbreviation("S2")
        'ét hai'
    """
    parts = []
    for ch in abbr:
        upper_ch = ch.upper()
        if upper_ch in ENGLISH_LETTER_PRONUNCIATION:
            parts.append(ENGLISH_LETTER_PRONUNCIATION[upper_ch])
        elif ch in DIGIT_PRONUNCIATION:
            parts.append(DIGIT_PRONUNCIATION[ch])
        elif ch == ' ':
            # Multi-word: space is a natural separator, just continue
            pass
    return " ".join(parts)
