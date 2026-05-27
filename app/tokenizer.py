import logging
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from typing import List
class LoggingEspeakTokenizer(EspeakTokenizer):
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        result = super().texts_to_tokens(texts)
        for text, tokens in zip(texts, result):
            logging.info(f"[Phoneme] '{text}' ? {' | '.join(tokens)}")
        return result