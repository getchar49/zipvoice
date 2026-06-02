import logging
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from typing import List
class LoggingEspeakTokenizer(EspeakTokenizer):
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        result = super().texts_to_tokens(texts)
        # Phoneme logging disabled — too verbose for production.
        # Uncomment for debugging tokenizer issues:
        # for text, tokens in zip(texts, result):
        #     logging.info(f"[Phoneme] '{text}' ? {' | '.join(tokens)}")
        return result