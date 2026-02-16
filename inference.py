import uuid
import logging
import importlib

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from config import settings
from base_chat_parser import BaseChatParser
from claude_schemas import (
    ClaudeMessage,
    ClaudeMessageParams,
    ClaudeTokenCount,
    ClaudeTokenCountParams,
    TextBlock,
    Usage,
)
from mlx_schemas import ChatParams
from server_sent_events import (
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
)

logger = logging.getLogger("uvicorn.error")

adapter = importlib.import_module(f"adapters.{settings.claude_mlx_adapter}")
claude_parser: BaseChatParser = adapter.Parser()


def load_llm(model_name: str, trust_remote_code: bool):
    logger.info(f"Loading MLX model: {model_name}")
    model, tokenizer = load(
        settings.model_name,
        tokenizer_config={"trust_remote_code": trust_remote_code},
    )
    logger.info("Model loaded successfully!")

    return model, tokenizer


def _finish_reason_to_stop_reason(finish_reason: str | None) -> str:
    """Map MLX finish_reason to Claude stop_reason."""
    logger.error(finish_reason)
    if finish_reason == "stop":
        return "end_turn"
    elif finish_reason == "length":
        return "max_tokens"
    return "end_turn"


async def claude_chat(model, tokenizer, chat: ChatParams):
    prompt = build_prompt(
        tokenizer,
        chat,
        add_generation_prompt=chat.add_generation_prompt,
        continue_final_message=chat.continue_final_message,
    )

    text = ""
    response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=make_sampler(**chat.sampler_params),
        max_tokens=chat.max_tokens,
    ):
        text += response.text

    prompt_tokens = response.prompt_tokens if response else len(prompt)
    generation_tokens = response.generation_tokens if response else 0
    stop_reason = _finish_reason_to_stop_reason(
        response.finish_reason if response else None
    )

    return ClaudeMessage(
        id=generate_response_id(),
        content=[TextBlock(type="text", text=text)],
        model=chat.request_model,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=prompt_tokens,
            output_tokens=generation_tokens,
        ),
    )


async def claude_chat_stream(model, tokenizer, chat: ChatParams):
    prompt = build_prompt(
        tokenizer,
        chat,
        add_generation_prompt=chat.add_generation_prompt,
        continue_final_message=chat.continue_final_message,
    )
    id = generate_response_id()

    # Initial usage with prompt tokens (will be updated at the end)
    initial_usage = {"input_tokens": 0, "output_tokens": 0}

    yield MessageStartEvent(id, chat.request_model, initial_usage).emit()
    yield ContentBlockStartEvent().emit()

    response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=make_sampler(**chat.sampler_params),
        max_tokens=chat.max_tokens,
    ):
        yield ContentBlockDeltaEvent("text_delta", response.text).emit()

    prompt_tokens = response.prompt_tokens if response else len(prompt)
    generation_tokens = response.generation_tokens if response else 0
    stop_reason = _finish_reason_to_stop_reason(
        response.finish_reason if response else None
    )
    final_usage = {
        "input_tokens": prompt_tokens,
        "output_tokens": generation_tokens,
    }

    yield ContentBlockStopEvent().emit()
    yield MessageDeltaEvent(stop_reason, usage=final_usage).emit()
    yield MessageStopEvent().emit()


def generate_response_id(prefix="msg"):
    return f"{prefix}_{uuid.uuid4().hex[:34]}"


def build_prompt(
    tokenizer,
    chat: ChatParams,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
):
    return tokenizer.apply_chat_template(
        chat.messages,
        tools=chat.tools,
        enable_thinking=chat.enable_thinking,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=True,
    )


def claude_tokens_count(tokenizer, params: ClaudeTokenCountParams) -> ClaudeTokenCount:
    message_params = ClaudeMessageParams(
        max_tokens=0,
        messages=params.messages,
        model=params.model,
        system=params.system,
        tools=params.tools,
        thinking=params.thinking,
        tool_choice=params.tool_choice,
    )
    chat = claude_parser.parse_chat_params(message_params)
    # Override to not add generation prompt for token counting
    tokens = build_prompt(
        tokenizer,
        chat,
        add_generation_prompt=False,
        continue_final_message=False,
    )
    return ClaudeTokenCount(input_tokens=len(tokens))
