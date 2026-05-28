"""
LLM-based normalizer client for special tokens (abbreviations, symbols, math, foreign words).

Sends individual tokens to an LLM API for context-aware normalization.
Falls back gracefully if the LLM is unavailable.
"""

import logging
import time
from functools import lru_cache
from typing import List, Tuple, Optional

import requests

from app.settings import LLM_API_URL, LLM_MODEL, LLM_API_KEY, LLM_TIMEOUT

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Bạn là công cụ chuẩn hóa văn bản cho hệ thống TTS tiếng Việt.

Bạn sẽ nhận một DANH SÁCH các mục cần chuẩn hóa, mỗi mục trên một dòng, có dạng:
[INDEX] nội_dung

Với mỗi mục, trả về dòng tương ứng:
[INDEX] kết_quả_chuẩn_hóa

QUY TẮC:
1. Từ viết tắt & chuỗi chữ cái (VD: AI, CEO, FCC, RoHS): Tách rời từng chữ cái, bọc MỖI chữ cái trong ngoặc nhọn < >, cách nhau bằng khoảng trắng.
   VD: AI -> <A> <I> ; CEO -> <C> <E> <O> ; RoHS -> <R> <o> <H> <S>
2. Chuỗi mã kết hợp chữ-số (VD: IEEE 802.11ax): Chữ cái bọc ngoặc nhọn, số đọc tách rời, dấu chấm đọc thành "chấm".
   VD: 802.11ax -> tám không hai chấm một một <a> <x>
3. Biểu thức toán học: Đọc thành lời tiếng Việt tự nhiên.
   VD: ax² + bx + c = 0 -> <a> <x> bình cộng <b> <x> cộng <c> bằng không
   VD: √(b²−4ac) -> căn bậc hai của <b> bình trừ bốn <a> <c>
   VD: Δ ≥ 0 -> đen ta lớn hơn hoặc bằng không
4. Ký hiệu đặc biệt đơn lẻ: Đọc thành tên tiếng Việt.
   VD: @ -> a còng ; # -> thăng ; $ -> đô la ; % -> phần trăm ; & -> và ; * -> sao
   VD: ± -> cộng trừ ; ≥ -> lớn hơn hoặc bằng ; ² -> bình phương
5. Từ/cụm ngoại ngữ: Giữ nguyên.
   VD: machine learning -> machine learning ; Wi-Fi -> Wi-Fi
6. TUYỆT ĐỐI KHÔNG thêm bớt, không giải thích, chỉ trả về kết quả theo đúng format [INDEX].
"""


class LLMNormalizer:
    """Client to normalize special tokens via LLM API."""

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

    def _call_llm(self, user_prompt: str) -> str:
        """Send a request to the LLM API with retry logic."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
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
