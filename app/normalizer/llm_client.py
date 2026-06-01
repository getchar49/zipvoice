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
Bạn là công cụ chuẩn hóa văn bản cho hệ thống chuyển văn bản thành giọng nói (TTS) tiếng Việt.

Bạn sẽ nhận một đoạn văn bản đã được chuẩn hóa sơ bộ bằng rule-based. Nhiệm vụ của bạn là chuẩn hóa hoàn chỉnh để TTS có thể đọc chính xác.

QUY TẮC:

1. TỪ VIẾT TẮT TIẾNG VIỆT:
   - Chuyển sang dạng đầy đủ hoặc phát âm tiếng Việt tự nhiên.
   - VD: UBND → ủy ban nhân dân, TP → thành phố, CSGT → cảnh sát giao thông

2. TỪ VIẾT TẮT TIẾNG ANH CẦN ĐỌC TỪNG CHỮ (spell out):
   - Đây là những từ viết tắt tiếng Anh mà cách đọc đúng là đánh vần từng chữ cái.
   - Bọc nguyên dạng gốc bằng marker {{SPELL}}...{{/SPELL}}
   - KHÔNG chuyển sang phát âm, giữ nguyên chữ gốc trong marker.
   - VD: TTS → {{SPELL}}TTS{{/SPELL}}, API → {{SPELL}}API{{/SPELL}}, CPU → {{SPELL}}CPU{{/SPELL}}
   - VD: USB → {{SPELL}}USB{{/SPELL}}, GPS → {{SPELL}}GPS{{/SPELL}}
   - CHÚ Ý: Một số từ viết tắt KHÔNG spell out mà đọc như từ (VD: NATO đọc "na tô", ASEAN đọc "a si an") → KHÔNG bọc marker, chuyển sang phát âm tiếng Việt bình thường.

3. TỪ NƯỚC NGOÀI (không phải viết tắt):
   - Chuyển sang cách phát âm tiếng Việt gần đúng.
   - VD: machine learning → mơ sin lơ ning, smartphone → sờ mát phôn, email → i meo

4. KÝ HIỆU ĐẶC BIỆT:
   - Chuyển sang tên đọc tiếng Việt.
   - VD: @ → a còng, # → thăng, & → và, % → phần trăm, $ → đô la

5. TOÁN HỌC VÀ KHOA HỌC:
   - Ký hiệu unicode toán học → đọc thành lời tiếng Việt.
   - Ký hiệu tiền tệ quốc tế → tên đơn vị + số.
   - Ký hiệu khoa học (×10^, v.v.) → đọc thành lời.

6. DẤU CÂU:
   - Chuẩn hóa TẤT CẢ dấu câu chỉ còn dấu CHẤM (.) và dấu PHẨY (,).
   - Dấu chấm hỏi (?) → dấu chấm (.)
   - Dấu chấm than (!) → dấu chấm (.)
   - Dấu chấm phẩy (;) → dấu phẩy (,)
   - Dấu hai chấm (:) → dấu phẩy (,)
   - Dấu ngoặc đơn/kép/vuông/nhọn → dấu chấm (.)
   - Dấu gạch ngang dài (—, –) → dấu phẩy (,)
   - Dấu ba chấm (...) → dấu chấm (.)
   - Dấu ngoặc kép/đơn ("", '', '') → bỏ hoặc thay bằng dấu chấm nếu ở ranh giới câu
   - Xuống dòng → dấu chấm (.)

7. QUY TẮC CHUNG:
   - GIỮ NGUYÊN phần text tiếng Việt đã chuẩn hóa đúng.
   - TUYỆT ĐỐI KHÔNG thêm nội dung mới, không giải thích, không comment.
   - Chỉ trả về đoạn văn bản đã chuẩn hóa hoàn chỉnh.
   - Đảm bảo output là text thuần (plain text), chỉ chứa chữ cái, số, dấu chấm, dấu phẩy, khoảng trắng, và marker {{SPELL}}...{{/SPELL}}.
"""


class LLMNormalizer:
    """Client to normalize text via LLM API.

    Supports two modes:
    - Token batch mode (legacy): normalize_tokens_batch()
    - Full-text mode (new): normalize_full_text()
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

    # ── Full-text normalization (new) ────────────────────────────────────

    def normalize_full_text(self, text: str) -> str:
        """
        Normalize an entire text passage via LLM in a single call.

        The LLM handles:
        - Vietnamese abbreviations → full form
        - English spell-out abbreviations → {{SPELL}}ABC{{/SPELL}} markers
        - Foreign words → Vietnamese phonetic approximation
        - Special symbols → Vietnamese names
        - Punctuation → only . and ,
        - Math/science notation → Vietnamese reading

        Args:
            text: Text already processed by rule-based normalizer.

        Returns:
            Fully normalized text with {{SPELL}} markers for abbreviations
            that need letter-by-letter pronunciation.
            Falls back to original text on LLM failure.
        """
        if not text or not text.strip():
            return text

        try:
            response_text = self._call_llm(
                user_prompt=text,
                system_prompt=FULL_TEXT_SYSTEM_PROMPT,
            )

            # Sanity checks
            if not response_text or not response_text.strip():
                logger.warning("LLM returned empty response, falling back to original")
                return text

            # Check that response isn't suspiciously different in length
            # (allow up to 3x expansion for abbreviation expansion)
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

    def _call_llm(self, user_prompt: str, system_prompt: str = None) -> str:
        """Send a request to the LLM API with retry logic."""
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
