import re
from nltk import sent_tokenize
from pathlib import Path
from app.normalizer.normalizer import TextNormalizer
from app.normalizer.abbre import ABBRE
import logging
from datetime import datetime

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
    text = normalizer.separate_comma_and_dot_at_the_end(text)
    # text = normalizer.separate_numbers_adjacent_chars(text)
    #print(f"text: {text}")
    text = normalizer.remove_urls(text)
    # print(f'0: {text}')
    text = normalizer.remove_emoji(text)
    text = normalizer.replace_special_words(text)
    # text = normalizer.remove_emoticons(text)
    # print(f'1: {text}')
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
    text = normalizer.norm_tag_verbatim(text)
    #print(f'22: {text}')
    text = normalizer.normalize_negative_number(text) 
    #print(f'23: {text}')
    text = normalizer.replace_dash_range(text)
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
    #text = text.replace('.', '. ')
    #text = text.replace('?', '? ')
    #text = text.replace(',', ', ')
    #print(f'27: {text}')
    text = re.sub(r'\.\s*', '. ', text)
    text = re.sub(r'\?\s*', '? ', text)
    text = re.sub(r',\s*', ', ', text)
    #print(f'28: {text}')
    text = normalizer.norm_duplicate_word(text)
    return text


def post_processing(text):
    text = re.sub('\s+', ' ', text)
    text = re.sub('\.\.\s', '. ', text)
    #text  = text.lower()
    return text

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

    # 4. Define the replacement logic
    def replace_match(match):
        # recover the matched word
        word = match.group(0)
        # return the value from dict using the lowercase key
        return my_dict.get(word.lower(), word)

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

def normalize_vietnamese_text(text):
    #logger.info(f"ORIGINAL INPUT: '{text}'")
    # Get the directory containing this file
    current_file_dir = Path(__file__).parent
    norm_text = []
    
    # Load English pronunciation dictionary
    dict = load_dict_english(str(current_file_dir / "english_word_3_v3.txt"))
    
    # Process each sentence
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
    # Apply English word pronunciations
    
    text = mapping_eng(text, dict)
    text = normalize_sentence_case(text)
    logger.info(f"FINAL OUTPUT: '{text}'")
    
    return text