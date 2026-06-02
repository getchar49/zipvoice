import re
from nltk import sent_tokenize
from pathlib import Path
from app.normalizer.normalizer import TextNormalizer
from app.normalizer.abbre import ABBRE
from app.normalizer.english_letters import spell_out_abbreviation
import logging
from datetime import datetime

from app.settings import USE_LLM_NORMALIZER, USE_DOUBLE_PUNCTUATION

def setup_logging():
    """Setup logging configuration for normalization monitoring"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = log_dir / f"normalization_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()


def normalize(text):
    normalizer = TextNormalizer()
    text = normalizer.norm_abbre(text, ABBRE)
    #print(f'abbre: {text}')
    # Remove URLs/emails BEFORE any punctuation processing (Bug 7,8)
    text = normalizer.remove_urls(text)
    # print(f'0: {text}')
    text = normalizer.separate_comma_and_dot_at_the_end(text)
    # text = normalizer.separate_numbers_adjacent_chars(text)
    #print(f"text: {text}")
    text = normalizer.remove_emoji(text)
    text = normalizer.replace_special_words(text)
    # text = normalizer.remove_emoticons(text)
    # print(f'1: {text}')
    # Convert verbatim symbols to words BEFORE removing special characters (Bug 6)
    text = normalizer.norm_tag_verbatim(text)
    #print(f'verbatim: {text}')
    text = normalizer.remove_special_characters_v1(text)
    #print(f'2: {text}')
    text = normalizer.normalize_number_plate(text)
    # print(f'3: {text}')
    text = normalizer.norm_ratio(text)
    # text = normalizer.norm_punct(text)
    # text = normalizer.separate_numbers_adjacent_chars(text)
    #print(f'4: {text}')
    text = normalizer.norm_unit(text)
    #print(f'5: {text}')
    text = normalizer.normalize_rate(text)
    #print(f'6: {text}')
    text = normalizer.norm_adress(text)
    #print(f'7: {text}')
    text = normalizer.norm_tag_fraction(text)
    #print(f'8: {text}')
    text = normalizer.normalize_phone_number(text)
    #print(f'9: {text}')
    text = normalizer.norm_multiply_number(text)
    #print(f'10: {text}')
    text = normalizer.normalize_sport_score(text)
    #print(f'11: {text}')
    text = normalizer.normalize_date_range(text)
    #print(f'12: {text}')
    text = normalizer.normalize_date(text)
    #print(f'13: {text}')
    text = normalizer.normalize_time_range(text)
    #print(f'14: {text}')
    text = normalizer.normalize_time(text)
    # print(f'15: {text}')
    # Now that time/phone/ratio are done, replace remaining colons (Bug 1)
    text = normalizer.norm_colon_to_period(text)
    text = normalizer.normalize_number_range(text)
    # print(f'16: {text}')
    text = normalizer.norm_id_digit(text)
    # print(f'17: {text}')
    text = normalizer.norm_soccer(text)
    #print(f'18: {text}')
    text = normalizer.norm_tag_roman_num(text)
    #print(f'19: {text}')
    text = normalizer.normalize_AZ09(text)
    # print(f'20: {text}')
    text = normalizer.norm_math_characters(text)
    # print(f'21: {text}')
    text = normalizer.normalize_negative_number(text) 
    #print(f'23: {text}')
    text = normalizer.replace_dash_range(text)
    text = normalizer.normalize_remaining_dash(text)
    text = normalizer.normalize_number(text)

    #text = text.replace('/', ' trên ')
    #print(f'24: {text}')
    text = normalizer.remove_special_characters_v2(text)
    #print(f'25: {text}')
    text = normalizer.norm_tag_roman_num(text)
    #print(f'26: {text}')
    text = normalizer.remove_multi_space(text)
    text = normalizer.normalize_number(text)

    #text = normalizer.lowercase(text)
    #print(f'27: {text}')
    # Fix punctuation spacing: ensure exactly one space after . ? ,
    # but ONLY when followed by a word character (avoids breaking `. ` at end)
    text = re.sub(r'\.(?=\S)', '. ', text)   # dot+nonspace → dot+space
    text = re.sub(r'\?(?=\S)', '? ', text)   # question+nonspace → question+space
    text = re.sub(r',(?=\S)', ', ', text)     # comma+nonspace → comma+space
    # Normalize multiple spaces after punctuation to exactly one
    text = re.sub(r'([.?!,])\s{2,}', r'\1 ', text)
    #print(f'28: {text}')
    text = normalizer.norm_duplicate_word(text)
    return text


def post_processing(text):
    text = re.sub('\s+', ' ', text)
    # Normalize 2+ consecutive dots to single dot + space
    # This runs BEFORE double punctuation, so single dots are preserved for that feature
    text = re.sub(r'\.{2,}\s*', '. ', text)
    #text  = text.lower()
    return text


def fix_punctuation_spacing(text):
    """
    Final cleanup: fix spaces around punctuation marks.
    The rule-based normalizer inserts ' word ' padded replacements which
    creates artifacts like ' . ' and ' , ' with extra leading spaces.
    """
    # Remove space BEFORE period/comma/semicolon/colon (but preserve 【】 brackets)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Deduplicate consecutive punctuation: ,, → , and ,. → . (period wins)
    text = re.sub(r'[,;:]+\.', '.', text)       # comma-like + period → period
    text = re.sub(r'\.[,;:]+', '.', text)       # period + comma-like → period
    text = re.sub(r',{2,}', ',', text)          # ,, → ,
    text = re.sub(r'\.{2,}', '.', text)         # .. → .
    # Ensure space AFTER period/comma/semicolon (when followed by a word char or bracket)
    text = re.sub(r'([.,;:!?])(?=[a-zA-Z\u00C0-\u1EF9【])', r'\1 ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def load_dict_english(path):
    with open(path, "r+", encoding="utf8") as f:
        lines = f.read().splitlines()
    rs = {}
    count = 0
    try:
        for line in lines:
            pair = line.split("|")
            rs[pair[0].lower()] = pair[1]
            count += 1
    except:
        print(count)
    return rs

def load_spell_out_words(path):
    """Load the spell-out abbreviation word list from file.
    
    Returns a set of uppercase abbreviation strings that should be
    spelled out letter-by-letter (e.g., {'TTS', 'API', 'CPU', ...}).
    """
    try:
        with open(path, "r", encoding="utf8") as f:
            words = set()
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    words.add(word.upper())
            return words
    except FileNotFoundError:
        logger.warning(f"Spell-out words file not found: {path}")
        return set()


def mapping_eng(text, my_dict):
    """Replace English words with Vietnamese phonetic pronunciations.
    
    Unlike the previous version, this does NOT wrap replacements in 【】 brackets.
    Words from english_word_3_v3.txt are read at normal TTS speed, not slow bracket speed.
    """
    if not text or not my_dict:
        return text

    # 1. Sort keys by length (Longest first).
    # This ensures "word sequence" is prioritized over "word".
    # e.g., if dict has 'new york' and 'new', we want to match 'new york' first.
    sorted_keys = sorted(my_dict.keys(), key=len, reverse=True)

    # 2. Escape keys to prevent regex errors (e.g., if a key is "c++")
    escaped_keys = [re.escape(k) for k in sorted_keys]

    # 3. Create a master pattern using OR (|).
    # \b matches a word boundary (start/end of word). 
    # This prevents matching 'c' inside 'Cách'.
    pattern_str = r'\b(' + '|'.join(escaped_keys) + r')\b'
    pattern = re.compile(pattern_str, re.IGNORECASE)

    # 4. Define the replacement logic — plain text replacement (no brackets)
    def replace_match(match):
        word = match.group(0)
        replacement = my_dict.get(word.lower(), word)
        return replacement

    # 5. Perform the substitution
    return pattern.sub(replace_match, text)


def normalize_sentence_case(text):
    if not text:
        return ""

    # 1. Capitalize the very first character of the string
    text = text[0].upper() + text[1:]

    # 2. Define a function to capitalize the captured group
    def capitalize_match(match):
        # match.group(1) is the punctuation + space (e.g., ". ")
        # match.group(2) is the letter to capitalize (e.g., "b")
        return match.group(1) + match.group(2).upper()

    # 3. Regex Pattern:
    # [.!?]  -> Match literal dot, exclamation, or question mark
    # \s+    -> Match one or more spaces
    # ([a-z])-> Capture the following lowercase letter
    pattern = r'([.!?]\s+)([a-z])'

    return re.sub(pattern, capitalize_match, text)


def apply_llm_normalization(text: str) -> str:
    """
    (LEGACY) Post-process text with LLM to normalize special tokens 
    token-by-token. Kept for backward compatibility.
    
    For the new pipeline, use apply_llm_full_text_normalization() instead.
    """
    from app.normalizer.special_token_detector import detect_special_tokens
    from app.normalizer.llm_client import get_llm_normalizer

    special_tokens = detect_special_tokens(text)
    if not special_tokens:
        logger.debug("No special tokens detected, skipping LLM normalization")
        return text

    logger.info(f"Detected {len(special_tokens)} special tokens for LLM normalization: "
                f"{[t.text for t in special_tokens[:10]]}{'...' if len(special_tokens) > 10 else ''}")

    llm = get_llm_normalizer()

    # Batch tokens (max 30 per batch to keep LLM token count small)
    MAX_BATCH = 30
    token_texts = [t.text for t in special_tokens]
    all_results = []

    for batch_start in range(0, len(token_texts), MAX_BATCH):
        batch = token_texts[batch_start:batch_start + MAX_BATCH]
        try:
            batch_results = llm.normalize_tokens_batch(batch)
            all_results.extend(batch_results)
        except Exception as e:
            logger.warning(f"LLM batch normalization failed: {e}. Keeping originals.")
            all_results.extend(batch)

    # Replace tokens in text (reverse order to preserve positions)
    result = text
    for token, normalized in reversed(list(zip(special_tokens, all_results))):
        if normalized and normalized != token.text:
            result = result[:token.start] + normalized + result[token.end:]
            logger.debug(f"LLM replaced: '{token.text}' -> '{normalized}'")

    return result


def apply_llm_full_text_normalization(text: str) -> str:
    """
    Normalize entire text via LLM in a single pass.
    
    The LLM handles:
    - Vietnamese/foreign abbreviations → Vietnamese pronunciation
    - Foreign words → Vietnamese pronunciation  
    - Special symbols → Vietnamese names
    - Punctuation → only . and ,
    - English spell-out abbreviations → {{SPELL}}ABC{{/SPELL}} markers
    
    Falls back to original text on any LLM failure.
    
    Args:
        text: Text already processed by rule-based normalizer + mapping_eng.
        
    Returns:
        Fully normalized text with {{SPELL}} markers for abbreviations.
    """
    from app.normalizer.llm_client import get_llm_normalizer
    
    llm = get_llm_normalizer()
    result = llm.normalize_full_text(text)
    
    logger.info(f"LLM full-text normalization result preview: '{result[:200]}...'")
    return result


def process_spell_out_markers(text: str, spell_out_set: set = None) -> str:
    """
    Process {{SPELL}}ABC{{/SPELL}} markers in text.
    
    For each marker:
    1. Extract the abbreviation (e.g., "TTS")
    2. Look up each letter in ENGLISH_LETTER_PRONUNCIATION
    3. Replace with a single 【pronunciation】 bracket group
    
    Also handles any remaining uppercase abbreviations that match
    the spell_out_set (supplementary file) but weren't caught by LLM.
    
    Args:
        text: Text containing {{SPELL}}...{{/SPELL}} markers from LLM.
        spell_out_set: Optional set of known spell-out words for fallback.
        
    Returns:
        Text with markers replaced by 【bracketed pronunciation】.
        
    Examples:
        "Hệ thống {{SPELL}}TTS{{/SPELL}} rất tốt"
        → "Hệ thống 【ti ti ét】 rất tốt"
    """
    # 1. Process {{SPELL}}...{{/SPELL}} markers from LLM
    # Match alphanumeric content (VVS1, S2) and multi-word (cron job)
    spell_pattern = re.compile(r'\{\{SPELL\}\}([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)\{\{/SPELL\}\}')
    
    def replace_spell_marker(match):
        abbr = match.group(1)
        pronunciation = spell_out_abbreviation(abbr)
        if pronunciation:
            logger.debug(f"Spell-out: '{abbr}' → '【{pronunciation}】'")
            return f'【{pronunciation}】'
        return abbr  # fallback: keep original if no pronunciation found
    
    text = spell_pattern.sub(replace_spell_marker, text)
    
    # 2. Fallback: check for remaining uppercase abbreviations in spell_out_set
    if spell_out_set:
        def replace_spell_out_word(match):
            word = match.group(0)
            if word.upper() in spell_out_set:
                pronunciation = spell_out_abbreviation(word)
                if pronunciation:
                    logger.debug(f"Spell-out (fallback): '{word}' → '【{pronunciation}】'")
                    return f'【{pronunciation}】'
            return word
        
        # Match uppercase words (2+ letters) that aren't already in brackets
        text = re.sub(r'(?<![【\w])\b[A-Z]{2,}\b(?![】\w])', replace_spell_out_word, text)
    
    return text


def apply_double_punctuation(text: str) -> str:
    """
    Experimental: Double all periods and commas for TTS pause testing.
    
    Replaces:
        , → ,,
        . → ..
        
    This is to test whether doubled punctuation produces better
    pauses/timing in the TTS audio output.
    
    Skips punctuation that's already doubled or inside 【】 brackets.
    """
    # Replace single periods (not already doubled, not inside brackets)
    # Use negative lookbehind/lookahead to avoid matching already-doubled
    text = re.sub(r'(?<!\.)\.(?!\.)', '..', text)
    text = re.sub(r'(?<!,),(?!,)', ',,', text)
    
    return text


def wrap_hardcoded_transliterations(text):
    """
    Wrap known transliterations from replace_special_words() in 【】 brackets.
    Disabled — replace_special_words hard-codes are now commented out.
    LLM full-text normalization handles these cases instead.
    """
    # Hard-coded mappings disabled — LLM handles these now
    # hardcoded = {
    #     r'\bây ai\b': '【ây ai】',
    #     r'\bki a\b': '【ki a】',
    #     r'\bai ti\b': '【ai ti】',
    # }
    # for pattern, replacement in hardcoded.items():
    #     text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_vietnamese_text(text):
    """
    Main entry point for normalizing Vietnamese text for TTS.
    
    Pipeline:
        Step 1: Rule-based normalize (dates, numbers, phones, abbreviations, etc.)
        Step 2: Post-processing cleanup
        Step 3: English word pronunciation mapping (NO brackets — normal speed)
        Step 4: LLM full-text normalization (if enabled)
                - Handles remaining abbreviations, foreign words, symbols, punctuation
                - Marks English spell-out abbreviations with {{SPELL}} markers
        Step 5: Process spell-out markers → 【bracketed pronunciation】
        Step 6: Double punctuation (experimental, if enabled)
        Step 7: Final punctuation cleanup
        Step 8: Sentence case normalization
    """
    logger.info(f"ORIGINAL INPUT: '{text}'")
    # Get the directory containing this file
    current_file_dir = Path(__file__).parent
    
    # Load English pronunciation dictionary
    dict = load_dict_english(str(current_file_dir / "english_word_3_v3.txt"))
    
    # Load spell-out abbreviation word list (supplementary file)
    spell_out_set = load_spell_out_words(str(current_file_dir / "spell_out_words.txt"))
    
    # Step 1: Rule-based normalize (existing pipeline)
    # Note: pre_normalize_special_formats() removed — LLM handles those cases now
    text = normalize(text)
    """
    for sentence in sent_tokenize(text):
        sentence = sentence.strip('.').strip()
        if not sentence:
            continue
        sentence = normalize(sentence)
        norm_text.append(sentence)
    
    text = '. '.join(norm_text)
    """
    
    # Step 2: Post-processing cleanup
    text = post_processing(text)
    logger.info(f"[Rule-based] Result: '{text}'")

    # Step 3: Apply English word pronunciations (NO brackets — normal speed)
    # Words from english_word_3_v3.txt are replaced inline without 【】 wrapping.
    # They will be read at normal TTS speed (speed=1, step=32).
    text = mapping_eng(text, dict)
    logger.info(f"[Mapping Eng] Result (LLM input): '{text}'")

    # Step 4: LLM full-text normalization
    # Handles: remaining abbreviations, foreign words, special symbols, punctuation
    # Marks spell-out abbreviations with {{SPELL}}ABC{{/SPELL}}
    if USE_LLM_NORMALIZER:
        text = apply_llm_full_text_normalization(text)
    
    # Step 5: Process {{SPELL}} markers → 【bracketed pronunciation】
    # Only these bracketed segments get slow speed (speed=0.4, step=64)
    text = process_spell_out_markers(text, spell_out_set)
    logger.info(f"[Spell-out] Result: '{text}'")

    # Step 6: Double punctuation (experimental)
    # , → ,,   . → ..
    if USE_DOUBLE_PUNCTUATION:
        text = apply_double_punctuation(text)

    # Step 7: Final punctuation cleanup
    text = fix_punctuation_spacing(text)

    # Step 8: Sentence case
    text = normalize_sentence_case(text)
    logger.info(f"FINAL OUTPUT: '{text}'")
    
    return text