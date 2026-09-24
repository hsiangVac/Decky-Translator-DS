# providers/deepseek_translate.py
"""DeepSeek OpenAI-compatible translation provider."""

import asyncio
import logging
from typing import List

import requests

from .base import TranslationProvider, ProviderType, NetworkError, ApiKeyError, RateLimitError

logger = logging.getLogger(__name__)


class DeepSeekTranslateProvider(TranslationProvider):
    ENDPOINT = "https://api.deepseek.com/chat/completions"
    SUPPORTED_LANGUAGES = [
        "auto", "en", "ja", "zh-CN", "zh-TW", "ko", "de", "fr", "es", "it",
        "pt", "ru", "ar", "nl", "no", "pl", "tr", "uk", "hi", "el", "th", "vi",
        "fi", "id", "ro", "bg", "hr", "cs", "hu", "sv", "da",
    ]
    LANGUAGE_NAMES = {
        "auto": "the detected source language", "en": "English", "ja": "Japanese",
        "zh-CN": "Simplified Chinese", "zh-TW": "Traditional Chinese", "ko": "Korean",
        "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
        "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "nl": "Dutch",
    }

    def __init__(self, api_key: str = "", model: str = "deepseek-chat"):
        self._api_key = api_key
        self._model = model

    def set_api_key(self, api_key: str) -> None:
        self._api_key = api_key

    def set_model(self, model: str) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "DeepSeek"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.DEEPSEEK

    def is_available(self, source_lang: str, target_lang: str) -> bool:
        return bool(self._api_key) and (source_lang == "auto" or source_lang in self.SUPPORTED_LANGUAGES) and target_lang in self.SUPPORTED_LANGUAGES

    def get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES.copy()

    def _request(self, texts: List[str], source_lang: str, target_lang: str):
        source = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target = self.LANGUAGE_NAMES.get(target_lang, target_lang)
        numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(texts))
        payload = {
            "model": self._model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": f"Translate each numbered item from {source} to {target}. Return only the translated items, one per line, preserving numbering and order."},
                {"role": "user", "content": numbered},
            ],
        }
        try:
            response = requests.post(self.ENDPOINT, json=payload, timeout=30, headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
        except requests.exceptions.ConnectionError as exc:
            raise NetworkError("No internet connection") from exc
        except requests.exceptions.Timeout as exc:
            raise NetworkError("DeepSeek request timed out") from exc
        if response.status_code in (401, 403):
            raise ApiKeyError("Invalid or unauthorized DeepSeek API key")
        if response.status_code == 429:
            raise RateLimitError("DeepSeek API rate limit exceeded")
        if response.status_code != 200:
            raise NetworkError(f"DeepSeek API error ({response.status_code}): {response.text[:200]}")
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise NetworkError("Unexpected response from DeepSeek API") from exc

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        result = await self.translate_batch([text], source_lang, target_lang)
        return result[0] if result else text

    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        if not texts:
            return []
        content = await asyncio.to_thread(self._request, texts, source_lang, target_lang)
        results = [""] * len(texts)
        for line in content.splitlines():
            line = line.strip()
            if ":" in line:
                index, value = line.split(":", 1)
                try:
                    i = int(index.strip())
                    if 0 <= i < len(results):
                        results[i] = value.strip()
                except ValueError:
                    pass
        return [value or original for value, original in zip(results, texts)]

    async def test_network(self) -> tuple:
        if not self._api_key:
            return False, "API key required"
        try:
            await self.translate("ok", "en", "zh-CN")
            return True, ""
        except (NetworkError, ApiKeyError, RateLimitError) as exc:
            return False, str(exc)
