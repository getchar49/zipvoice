"""
Persistent dictionary cache for English-to-Vietnamese phonetic pairs.

Stores pairs learned from LLM responses across runs.
On each new text, known words are replaced first (pre-LLM lookup),
reducing the amount of work the LLM needs to do.

File format: JSON object mapping lowercase English words to Vietnamese phonetic spellings.
Example: {"subscription": "xắp scrip sần", "server": "sơ vơ"}
"""
import json
import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DICT_PATH = Path(__file__).parent / "learned_dict.json"


class DictionaryCache:
    """Thread-safe persistent dictionary for English→Vietnamese phonetic pairs."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else DEFAULT_DICT_PATH
        self._lock = threading.Lock()
        self._dict: Dict[str, str] = {}
        self._pattern: Optional[re.Pattern] = None
        self._load()

    def _load(self):
        """Load dictionary from disk."""
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._dict = json.load(f)
                self._rebuild_pattern()
                logger.info(f"Loaded {len(self._dict)} learned pairs from {self.path}")
            except Exception as e:
                logger.warning(f"Failed to load dictionary cache: {e}")
                self._dict = {}

    def _save(self):
        """Persist dictionary to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self._dict, f, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception as e:
            logger.warning(f"Failed to save dictionary cache: {e}")

    def _rebuild_pattern(self):
        """Rebuild the compiled regex pattern from current dictionary keys."""
        if not self._dict:
            self._pattern = None
            return
        # Sort keys by length (longest first) to match multi-word phrases first
        sorted_keys = sorted(self._dict.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]
        pattern_str = r'\b(' + '|'.join(escaped_keys) + r')\b'
        self._pattern = re.compile(pattern_str, re.IGNORECASE)

    def lookup(self, word: str) -> Optional[str]:
        """Look up a single word in the dictionary."""
        return self._dict.get(word.lower())

    def update_pairs(self, pairs: List[dict]):
        """
        Add new English-Vietnamese pairs to the dictionary.

        Args:
            pairs: List of dicts with keys "english_word" and "vietnamese_spelling".
        """
        with self._lock:
            added = 0
            for pair in pairs:
                eng = pair.get("english_word", "").lower().strip()
                vie = pair.get("vietnamese_spelling", "").strip()
                if eng and vie and eng not in self._dict:
                    self._dict[eng] = vie
                    added += 1
            if added > 0:
                self._rebuild_pattern()
                self._save()
                logger.info(f"Dictionary updated: +{added} new pairs (total: {len(self._dict)})")

    def apply_to_text(self, text: str) -> str:
        """
        Replace known English words in text with Vietnamese phonetic spelling.

        Uses pre-compiled regex pattern for efficient matching.
        Matches are case-insensitive with word boundaries.
        """
        if not self._dict or not self._pattern:
            return text

        def replace_match(match):
            word = match.group(0).lower()
            return self._dict.get(word, match.group(0))

        return self._pattern.sub(replace_match, text)

    @property
    def size(self) -> int:
        """Number of entries in the dictionary."""
        return len(self._dict)


# Module-level singleton — lazy init
_instance: Optional[DictionaryCache] = None


def get_dictionary_cache() -> DictionaryCache:
    """Get or create the singleton DictionaryCache instance."""
    global _instance
    if _instance is None:
        _instance = DictionaryCache()
    return _instance
