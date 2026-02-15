from abc import ABC, abstractmethod
from typing import Any, Dict, List

from mlx_schemas import ChatParams


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
    def format_response(self, raw_output: str) -> Dict[str, Any]:
        """
        Cleans the model's output (stripping Markdown, thinking tags, etc.)
        and wraps it back into a Claude-compatible JSON response.
        """
        pass
