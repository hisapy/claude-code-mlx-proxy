from abc import ABC, abstractmethod
from typing import Any, Dict

from mlx_schemas import ChatParams
from claude_schemas import ContentBlock


class StreamTextSanitizer:
    """Stateful sanitizer for streamed model text chunks.

    Implementations can buffer partial markers/tags across chunk boundaries
    and emit only safe user-facing text from `push()`, then flush remaining
    sanitized text in `finish()`.

    Example: `adapters/qwen3.py` implements `_Qwen3StreamSanitizer` to strip
    split `<think>` and chat-template markers such as `<|im_start|>` / `<|im_end|>`
    while streaming.
    """

    def push(self, chunk: str) -> str:
        return chunk

    def finish(self) -> str:
        return ""


class BaseChatParser(ABC):
    """
    Interface for LLM-specific message formatting.
    You might need an specific subclass in an adapter depending on the model (Qwen3, DeepSeek, etc.)
    """

    @abstractmethod
    def parse_chat_params(self, params: Dict[str, Any]) -> ChatParams:
        """
        Standardizes the incoming 'messages' from Claude Code into
        the format expected by the specific model's tokenizer.
        """
        pass

    @abstractmethod
    def parse_response_text(self, text: str) -> ContentBlock:
        """
        Cleans the model's output (stripping Markdown, thinking tags, etc.)
        and wraps it back into a Claude-compatible JSON response.
        """
        pass

    def sanitize_response_text(self, text: str) -> str:
        return text

    def create_stream_text_sanitizer(self) -> StreamTextSanitizer:
        return StreamTextSanitizer()

    def default_stop_sequences(self) -> list[str]:
        return []
