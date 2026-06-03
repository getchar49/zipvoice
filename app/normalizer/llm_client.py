"""
LLM-based normalizer client for TTS text normalization.

Supports two modes:
1. Token-by-token normalization (legacy) — normalize_tokens_batch()
2. Full-text normalization (new) — normalize_full_text()

The full-text mode sends the entire pre-processed text to the LLM,
which handles abbreviations, foreign words, special symbols, and punctuation
in a single pass.

Falls back gracefully if the LLM is unavailable.
"""

import logging
import re
import time
import concurrent.futures
import json
from functools import lru_cache
from typing import List, Tuple, Optional, Dict

import requests

from app.settings import LLM_API_URL, LLM_MODEL, LLM_API_KEY, LLM_TIMEOUT

logger = logging.getLogger(__name__)

# ── Legacy prompt (token-by-token mode) ──────────────────────────────────

SYSTEM_PROMPT = """\
Bạn là công cụ chuẩn hóa văn bản cho hệ thống TTS tiếng Việt.

Bạn sẽ nhận một DANH SÁCH các mục cần chuẩn hóa, mỗi mục trên một dòng, có dạng:
[INDEX] nội_dung

Với mỗi mục, trả về dòng tương ứng:
[INDEX] kết_quả_chuẩn_hóa

QUY TẮC:
1. Từ viết tắt & chuỗi chữ cái (VD: AI, CEO, FCC, RoHS): Tách rời từng chữ cái, bọc MỖI chữ cái trong ngoặc 【 】, cách nhau bằng khoảng trắng.
   VD: AI -> 【A】 【I】 ; CEO -> 【C】 【E】 【O】 ; RoHS -> 【R】 【o】 【H】 【S】
2. Chuỗi mã kết hợp chữ-số (VD: IEEE 802.11ax): Chữ cái bọc ngoặc 【】, số đọc tách rời, dấu chấm đọc thành "chấm".
   VD: 802.11ax -> tám không hai chấm một một 【a】 【x】
3. Biểu thức toán học: Đọc thành lời tiếng Việt tự nhiên.
   VD: ax² + bx + c = 0 -> 【a】 【x】 bình cộng 【b】 【x】 cộng 【c】 bằng không
   VD: √(b²−4ac) -> căn bậc hai của 【b】 bình trừ bốn 【a】 【c】
   VD: Δ ≥ 0 -> đen ta lớn hơn hoặc bằng không
4. Ký hiệu đặc biệt đơn lẻ: Đọc thành tên tiếng Việt.
   VD: @ -> a còng ; # -> thăng ; $ -> đô la ; % -> phần trăm ; & -> và ; * -> sao
   VD: ± -> cộng trừ ; ≥ -> lớn hơn hoặc bằng ; ² -> bình phương
5. Từ/cụm ngoại ngữ: Giữ nguyên.
   VD: machine learning -> machine learning ; Wi-Fi -> Wi-Fi
6. TUYỆT ĐỐI KHÔNG thêm bớt, không giải thích, chỉ trả về kết quả theo đúng format [INDEX].
"""

# ── New prompt (full-text mode) ──────────────────────────────────────────

FULL_TEXT_SYSTEM_PROMPT = """\
Bạn là công cụ chuẩn hóa văn bản cho hệ thống TTS tiếng Việt.

QUY TẮC:

1. VIẾT TẮT TIẾNG VIỆT → dạng đầy đủ.
   VD: UBND → ủy ban nhân dân, TP → thành phố

2. VIẾT TẮT / CHỮ CÁI TIẾNG ANH → bọc {{SPELL}}...{{/SPELL}}.
   Áp dụng cho: chữ hoa ≤5 ký tự không đọc được như từ, HOẶC chữ cái đơn lẻ viết hoa.
   VD: TTS → {{SPELL}}TTS{{/SPELL}}, AI → {{SPELL}}AI{{/SPELL}}, VS1 → {{SPELL}}VS1{{/SPELL}}, G → {{SPELL}}G{{/SPELL}}, S2 → {{SPELL}}S2{{/SPELL}}
   Ngoại lệ đọc như từ: NATO → na tô, ASEAN → a si an.

3. TỪ TIẾNG ANH ĐẦY ĐỦ → phát âm tiếng Việt gần đúng. KHÔNG dùng {{SPELL}}.
   VD: carat → ca rát, server → sơ vơ, test → tét, file → phai, online → on lai

4. KÝ HIỆU → tên tiếng Việt.
   VD: @ → a còng, # → thăng, & → và, $ → đô la

5. DẤU CÂU → chỉ giữ dấu chấm (.) và phẩy (,).

6. TUYỆT ĐỐI KHÔNG:
   - KHÔNG thay đổi nội dung gốc. KHÔNG thêm/bớt/sửa chữ cái hay số. VD: VS1 giữ nguyên VS1, KHÔNG đổi thành VVS1.
   - KHÔNG đoán ngược viết tắt từ phiên âm. VD: nếu input là "vê ét một" thì giữ nguyên, KHÔNG đổi thành VVS1 hay VS1.
   - KHÔNG giải thích.
   - Output chỉ chứa: chữ cái, số, dấu chấm, phẩy, khoảng trắng, {{SPELL}}...{{/SPELL}}.
"""

# ── Extraction prompt (structured output mode) ───────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert phonetician specializing in Foreign-Language-to-Vietnamese transliteration (Việt hóa). Your task is to analyze the user's text and perform two strict operations:
1. INITIALISM EXTRACTION (SPELLED-OUT ABBREVIATIONS ONLY)
Scan the text and extract true initialisms (e.g., AWS, EC2, CI/CD, API, CEO). 
- ONLY extract abbreviations where the letters are pronounced individually one-by-one (e.g., A-W-S, A-P-I).
- STRICT EXCLUSION - PRONOUNCEABLE ACRONYMS: Do NOT extract abbreviations that are pronounced as fluid words (e.g., ignore "NASA", "BERT", "SCADA", "YOLO"). These must be routed to the Phonetic Spelling section.
- STRICT EXCLUSION - PROPER NOUNS: Do not extract proper nouns, brand names, or software products from the technical lexicon (e.g., ignore "GitHub", "Actions", "Docker", "Kubernetes", "Linux").
- Format rule: Initialisms can be lowercase or uppercase (api, CTO, CEO), they may have specific technical formats with symbols/numbers (e.g., "CI/CD", "EC2", "K8s"). 
- Reject any standard Title Case or PascalCase words (e.g., reject "GitHub", reject "Actions").

2. PHONETIC VIETNAMESE SPELLING (VIỆT HÓA)
Extract ONLY the foreign (non-Vietnamese) words AND pronounceable acronyms from the text and convert them into how they would be spelled phonetically using the Vietnamese alphabet.
- PROPER NOUN OVERRIDE: You MUST extract foreign proper nouns, brand names, company names, and software products (e.g., "Docker", "Linux", "Kubernetes", "GitHub") and convert them into phonetic Vietnamese spelling. Do not skip them.
- PRONOUNCEABLE ACRONYMS: Include abbreviations here if they are spoken as a word rather than spelled out (e.g., "NASA" -> "na xa", "BERT" -> "bớt").
- STRICT LANGUAGE FILTER: Completely ignore any words that are already in Vietnamese (e.g., "kết", "nối", "giám", "sát"). Do not include them in the JSON output under any circumstances.
- DO NOT translate the meaning of the word. You are spelling out how the foreign word SOUNDS in its native language using Vietnamese phonetics.
- DO NOT use hyphens for multi-syllable words ("mơ xin", "đi zi tồ") under any circumstances.

Examples of the required phonetic conversion across different languages, acronyms, and tech brands:
- [English] "production" -> "pờ rô đắc sừn"
- [English] "standard" -> "xờ ten đợt"
- [French] "croissant" -> "cờ roa sòn"
- [Japanese] "sushi" -> "su xi"
- [Russian] "vodka" -> "vốt ca"
- [Acronym] "NASA" -> "na xa"
- [Acronym] "BERT" -> "bớt"
- [Acronym] "SCADA" -> "xca đa"
- [Tech Brand] "Docker" -> "đốc cơ"
- [Tech Brand] "Kubernetes" -> "ku bơ nét"
- [Tech Brand] "Linux" -> "li nắc"
- [Tech Brand] "GitHub" -> "gít hắp"

OUTPUT FORMAT
You must return the results exactly according to the provided JSON schema. Do not include any markdown formatting, conversational filler, or explanations.
"""

EXTRACTION_RESPONSE_FORMAT = {
            "type": "json_schema",
            "json_schema": {
                "name": "acronyms_extraction_and_convert_to_viet_spelling",
                "strict": True,
                "schema": {
                "type": "object",
                "properties": {
                    "acronyms": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of acronyms extracted from the text (e.g., WHO, NASA)."
                    },
                    "foreign_vietnamese_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                        "foreign_word": { "type": "string" },
                        "vietnamese_spelling": { "type": "string" }
                        },
                        "required": ["foreign_word", "vietnamese_spelling"],
                        "additionalProperties": False
                    },
                    "description": "Pairs of Foreign words from the text and their Vietnamese spelling."
                    }
                },
                "required": ["acronyms", "foreign_vietnamese_pairs"],
                "additionalProperties": False
                }
            }
        }



class LLMNormalizer:
    """Client to normalize text via LLM API.

    Supports three modes:
    - Token batch mode (legacy): normalize_tokens_batch()
    - Full-text mode (legacy v2): normalize_full_text()
    - Structured extraction mode (current): extract_acronyms_and_pairs()
    """

    def __init__(
        self,
        api_url: str = LLM_API_URL,
        model: str = LLM_MODEL,
        api_key: str = LLM_API_KEY,
        timeout: int = LLM_TIMEOUT,
        max_retries: int = 2,
    ):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # In-memory cache: token_string -> normalized_string
        self._cache: dict = {}

    def _chunk_text(self, text: str, max_chars: int = 2500) -> List[str]:
        """
        Chia văn bản thành các chunk nhỏ dựa trên giới hạn max_chars.
        Ưu tiên cắt theo đoạn văn, sau đó đến câu, và cuối cùng là từ
        để không làm đứt gãy ngữ nghĩa hoặc cắt đôi một từ/viết tắt.
        """
        if len(text) <= max_chars:
            return [text]

        # Dùng regex để split nhưng vẫn giữ lại các ký tự phân cách
        paragraphs = re.split(r'(\n+)', text)
        chunks = []
        current_chunk = ""

        def add_to_chunks(segment):
            nonlocal current_chunk
            # Nếu thêm segment vào làm vượt quá max_chars và chunk hiện tại đã có nội dung
            if len(current_chunk) + len(segment) > max_chars and current_chunk.strip():
                chunks.append(current_chunk)
                current_chunk = segment
            else:
                current_chunk += segment

        for p in paragraphs:
            if len(p) > max_chars:
                # Nếu 1 đoạn văn quá dài, tiếp tục cắt theo câu
                sentences = re.split(r'([.?!;:]\s+)', p)
                for s in sentences:
                    if len(s) > max_chars:
                        # Nếu 1 câu quá dài, cắt theo khoảng trắng (từ)
                        words = re.split(r'(\s+)', s)
                        for w in words:
                            add_to_chunks(w)
                    else:
                        add_to_chunks(s)
            else:
                add_to_chunks(p)

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks
    # ── Structured extraction (current) ────────────────────────────────

    def extract_acronyms_and_pairs(self, text: str, max_chars_per_chunk: int = 2500) -> dict:
        """
        Extract acronyms and foreign-Vietnamese phonetic pairs from text
        using LLM with structured output (response_format).

        Args:
            text: Text already processed by rule-based normalizer + mapping_eng.

        Returns:
            Dict with keys:
                - "acronyms": List[str] — acronyms found in text
                - "foreign_vietnamese_pairs": List[dict] — each with
                  "foreign_word" and "vietnamese_spelling"
            Returns empty result on LLM failure.
        """
        empty_result = {"acronyms": [], "foreign_vietnamese_pairs": []}

        if not text or not text.strip():
            return empty_result

        # 1. Chia text thành các chunk an toàn
        chunks = self._chunk_text(text, max_chars=max_chars_per_chunk)
        
        # Biến lưu trữ kết quả để khử trùng lặp
        all_acronyms = set()
        all_pairs_dict = {} # key: foreign_word, value: vietnamese_spelling

        # Hàm xử lý cho từng chunk
        def process_chunk(chunk_text: str) -> dict:
            if not chunk_text.strip():
                return empty_result
            try:
                response_text = self._call_llm(
                    user_prompt=chunk_text,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    response_format=EXTRACTION_RESPONSE_FORMAT,
                )
                if not response_text or not response_text.strip():
                    return empty_result
                    
                result = json.loads(response_text.strip())
                if not isinstance(result, dict):
                    return empty_result
                return result
            except Exception as e:
                logger.warning(f"LLM extraction failed for chunk, skipping. Error: {e}")
                return empty_result

        # 2. Xử lý song song các chunks (Max 5 workers hoặc bằng số lượng chunks để không spam API quá mức)
        max_workers = min(5, max(1, len(chunks)))
        logger.info(f"Splitting text into {len(chunks)} chunks, processing with {max_workers} workers.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))

        # 3. Gộp kết quả và khử trùng lặp
        for res in chunk_results:
            # Gộp Acronyms
            acronyms = res.get("acronyms", [])
            if isinstance(acronyms, list):
                for acr in acronyms:
                    if isinstance(acr, str) and acr.strip():
                        all_acronyms.add(acr.strip())
            
            # Gộp Pairs
            pairs = res.get("foreign_vietnamese_pairs", [])
            if isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, dict):
                        fw = pair.get("foreign_word")
                        vs = pair.get("vietnamese_spelling")
                        if fw and vs and isinstance(fw, str) and isinstance(vs, str):
                            # Nếu từ ngoại ngữ xuất hiện nhiều lần, ta giữ bản dịch đầu tiên hoặc ghi đè (ở đây là ghi đè)
                            all_pairs_dict[fw.strip()] = vs.strip()

        final_acronyms = sorted(list(all_acronyms))
        final_pairs = [{"foreign_word": k, "vietnamese_spelling": v} for k, v in all_pairs_dict.items()]

        logger.info(f"LLM extraction complete: {len(final_acronyms)} acronyms, {len(final_pairs)} pairs extracted.")
        
        return {
            "acronyms": final_acronyms,
            "foreign_vietnamese_pairs": final_pairs
        }

    # ── Full-text normalization (legacy v2) ───────────────────────────────

    def normalize_full_text(self, text: str) -> str:
        """
        (LEGACY) Normalize entire text via LLM in a single pass.
        Kept for backward compatibility.
        """
        if not text or not text.strip():
            return text

        try:
            response_text = self._call_llm(
                user_prompt=text,
                system_prompt=FULL_TEXT_SYSTEM_PROMPT,
            )

            if not response_text or not response_text.strip():
                logger.warning("LLM returned empty response, falling back to original")
                return text

            if len(response_text) > len(text) * 3:
                logger.warning(
                    f"LLM response suspiciously long ({len(response_text)} vs {len(text)}), "
                    f"falling back to original"
                )
                return text

            logger.info(f"LLM full-text normalization completed. "
                        f"Input: {len(text)} chars, Output: {len(response_text)} chars")
            return response_text.strip()

        except Exception as e:
            logger.warning(f"LLM full-text normalization failed, falling back to original: {e}")
            return text

    # ── Token batch normalization (legacy) ───────────────────────────────

    def normalize_token(self, token: str) -> str:
        """Normalize a single special token via LLM. Returns original on failure."""
        results = self.normalize_tokens_batch([token])
        return results[0]

    def normalize_tokens_batch(self, tokens: List[str]) -> List[str]:
        """
        Normalize a batch of special tokens in a single LLM call.

        Sends all tokens as a numbered list, parses numbered results back.
        Falls back to returning originals if LLM is unavailable.
        """
        if not tokens:
            return []

        # Check cache first — only send uncached tokens to LLM
        results = [None] * len(tokens)
        uncached_indices = []
        for i, token in enumerate(tokens):
            cached = self._cache.get(token)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return results  # All cached

        # Build prompt for uncached tokens
        prompt_lines = []
        for idx, orig_idx in enumerate(uncached_indices):
            prompt_lines.append(f"[{idx}] {tokens[orig_idx]}")
        user_prompt = "\n".join(prompt_lines)

        # Call LLM
        try:
            response_text = self._call_llm(user_prompt)
            parsed = self._parse_batch_response(response_text, len(uncached_indices))

            for idx, orig_idx in enumerate(uncached_indices):
                normalized = parsed.get(idx, tokens[orig_idx])
                # Sanity check: if LLM returned empty or suspiciously long, fallback
                if not normalized.strip() or len(normalized) > len(tokens[orig_idx]) * 10:
                    normalized = tokens[orig_idx]
                results[orig_idx] = normalized
                self._cache[tokens[orig_idx]] = normalized

        except Exception as e:
            logger.warning(f"LLM normalization failed, falling back to originals: {e}")
            for orig_idx in uncached_indices:
                results[orig_idx] = tokens[orig_idx]

        return results

    # ── LLM API call ─────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str, system_prompt: str = None,
                  response_format: dict = None) -> str:
        """Send a request to the LLM API with retry logic.
        
        Args:
            user_prompt: The user message content.
            system_prompt: System prompt (defaults to SYSTEM_PROMPT).
            response_format: Optional response_format dict for structured output.
        """
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug(f"LLM response: {content}")
                return content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 0.5 * (2 ** attempt)  # exponential backoff: 0.5s, 1s
                    logger.warning(f"LLM call attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)

        raise last_error

    def _parse_batch_response(self, response_text: str, expected_count: int) -> dict:
        """
        Parse LLM response in format:
            [0] result_0
            [1] result_1
        Returns dict {index: result_string}.
        """
        parsed = {}
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Try to match [N] pattern
            if line.startswith("["):
                bracket_end = line.find("]")
                if bracket_end > 0:
                    try:
                        idx = int(line[1:bracket_end])
                        value = line[bracket_end + 1:].strip()
                        parsed[idx] = value
                    except ValueError:
                        continue
        return parsed

    def clear_cache(self):
        """Clear the in-memory normalization cache."""
        self._cache.clear()


# Module-level singleton — lazy init
_llm_normalizer_instance: Optional[LLMNormalizer] = None


def get_llm_normalizer() -> LLMNormalizer:
    """Get or create the singleton LLMNormalizer instance."""
    global _llm_normalizer_instance
    if _llm_normalizer_instance is None:
        _llm_normalizer_instance = LLMNormalizer()
    return _llm_normalizer_instance
