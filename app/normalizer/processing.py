import re
from nltk import sent_tokenize
from pathlib import Path
from app.normalizer.normalizer import TextNormalizer
from app.normalizer.abbre import ABBRE
from app.normalizer.special_token_detector import detect_special_tokens
import logging
from datetime import datetime

from app.settings import USE_LLM_NORMALIZER

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

def pre_normalize_special_formats(text):
    """
    Pre-process text to handle formats that the existing rule-based normalizer
    doesn't handle well. Runs BEFORE the main normalize() pipeline.
    """
    # 1. Unicode math operators → ASCII equivalents (so downstream handlers work)
    text = text.replace('×', 'x')   # multiplication sign
    text = text.replace('÷', '/')   # division sign
    text = text.replace('−', '-')   # minus sign (unicode) → hyphen-minus
    text = text.replace('≤', '<=')
    text = text.replace('≥', '>=')
    text = text.replace('≠', '!=')
    text = text.replace('±', '+-')

    # 2. Scientific notation: 3.5×10^6 → ba chấm năm nhân mười mũ sáu
    # (After × → x above, this becomes 3.5x10^6)
    def sci_notation_to_words(m):
        base = m.group(1)       # e.g. "3.5" or "3"
        exponent = m.group(2)   # e.g. "6"
        return f'{base} nhân mười mũ {exponent}'
    text = re.sub(r'(\d+(?:[.,]\d+)?)\s*[xX]\s*10\^(\d+)', sci_notation_to_words, text)

    # 3. Currency: $1,234.5678 (international format with comma as thousand sep)
    # Convert to a form the VN normalizer can handle
    def currency_to_words(m):
        symbol = m.group(1)     # e.g. "$" or "€"
        number = m.group(2)     # e.g. "1,234.5678" or "0.000045"
        currency_names = {'$': 'đô la', '€': 'ơ rô', '£': 'bảng', '¥': 'yên'}
        currency_name = currency_names.get(symbol, symbol)
        # Remove thousand separators (commas in international format)
        # but keep the decimal point
        clean_number = number.replace(',', '')
        return f'{currency_name} {clean_number}'
    text = re.sub(r'([\$€£¥])(\d[\d,]*(?:\.\d+)?)', currency_to_words, text)

    return text


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
    text = re.sub('\.\.\s', '. ', text)
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

def mapping_eng(text, my_dict):
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

    # 4. Define the replacement logic — wrap transliterations in 【】 brackets
    # so bracket-aware inference uses slower speed for better pronunciation
    def replace_match(match):
        word = match.group(0)
        replacement = my_dict.get(word.lower(), word)
        # Wrap in brackets: "ây ai" -> "【ây ai】"
        return f'【{replacement}】'

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
    Post-process text with LLM to normalize special tokens that rule-based
    normalizer couldn't handle well (abbreviations, symbols, math, foreign words).

    Detects special tokens, batches them, sends to LLM, and replaces in text.
    Falls back to original text on any LLM failure.

    Args:
        text: Text already processed by rule-based normalizer.

    Returns:
        Text with special tokens normalized by LLM (may contain <X> markers).
    """
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


def wrap_hardcoded_transliterations(text):
    """
    Wrap known transliterations from replace_special_words() in 【】 brackets.
    These were converted early in the pipeline before brackets could be added.
    """
    # Map of transliterations created by replace_special_words
    hardcoded = {
        r'\bây ai\b': '【ây ai】',
        r'\bki a\b': '【ki a】',
        r'\bai ti\b': '【ai ti】',
    }
    for pattern, replacement in hardcoded.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_vietnamese_text(text):
    #logger.info(f"ORIGINAL INPUT: '{text}'")
    # Get the directory containing this file
    current_file_dir = Path(__file__).parent
    norm_text = []
    
    # Load English pronunciation dictionary
    dict = load_dict_english(str(current_file_dir / "english_word_3_v3.txt"))
    
    # Step 0: Pre-process special formats (currency, scientific notation, unicode math)
    text = pre_normalize_special_formats(text)
    
    # Step 1: Rule-based normalize (existing pipeline)
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
    #print(f'29: {text}')
    text = post_processing(text)
    #print(f'30: {text}')

    # Step 2: LLM post-processing for special tokens
    # Must run BEFORE mapping_eng so that bracket markers from English
    # transliterations don't get detected as residual special characters.
    if USE_LLM_NORMALIZER:
        text = apply_llm_normalization(text)

    # Step 3: Apply English word pronunciations (creates 【..】 bracket markers)
    text = mapping_eng(text, dict)

    # Wrap hardcoded transliterations from replace_special_words in 【】 brackets
    # These were transliterated early in the pipeline (before we could wrap them)
    text = wrap_hardcoded_transliterations(text)

    # Step 4: Final punctuation cleanup
    text = fix_punctuation_spacing(text)

    text = normalize_sentence_case(text)
    logger.info(f"FINAL OUTPUT: '{text}'")
    
    return text