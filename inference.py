import uuid
import logging
import importlib
import time

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


def _resolve_prompt_tokens(last_response, prompt):
    return last_response.prompt_tokens if last_response else len(prompt)


def _resolve_generation_tokens(last_response):
    return last_response.generation_tokens if last_response else 0


def _resolve_stop_reason(last_response):
    finish_reason = last_response.finish_reason if last_response else None
    return _finish_reason_to_stop_reason(finish_reason)


def _format_dialog(chat: ChatParams) -> str:
    lines = []
    for message in chat.messages:
        role = message.get("role", "unknown")
        content = str(message.get("content", ""))
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _log_incoming_dialog(chat: ChatParams):
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Incoming conversation\n%s",
        _format_dialog(chat),
    )


def _log_generation_summary(
    *,
    generated_text: str,
    last_response,
    elapsed_seconds: float,
    is_streaming: bool,
):
    if not logger.isEnabledFor(logging.DEBUG):
        return

    generation_tokens = _resolve_generation_tokens(last_response)
    measured_tps = generation_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

    logger.debug(
        "Generation summary (%s) elapsed=%.3fs prompt_tokens=%s generation_tokens=%s "
        "finish_reason=%s prompt_tps=%s generation_tps=%s measured_tps=%.2f peak_memory_gb=%s",
        "stream" if is_streaming else "non-stream",
        elapsed_seconds,
        getattr(last_response, "prompt_tokens", None),
        getattr(last_response, "generation_tokens", None),
        getattr(last_response, "finish_reason", None),
        getattr(last_response, "prompt_tps", None),
        getattr(last_response, "generation_tps", None),
        measured_tps,
        getattr(last_response, "peak_memory", None),
    )
    logger.debug("Generated response\n[assistant] %s", generated_text)


async def claude_chat(model, tokenizer, chat: ChatParams):
    _log_incoming_dialog(chat)

    prompt = build_prompt(
        tokenizer,
        chat,
        add_generation_prompt=chat.add_generation_prompt,
        continue_final_message=chat.continue_final_message,
    )

    start_time = time.perf_counter()
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

    prompt_tokens = _resolve_prompt_tokens(response, prompt)
    generation_tokens = _resolve_generation_tokens(response)
    stop_reason = _resolve_stop_reason(response)

    _log_generation_summary(
        generated_text=text,
        last_response=response,
        elapsed_seconds=time.perf_counter() - start_time,
        is_streaming=False,
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
    _log_incoming_dialog(chat)

    prompt = build_prompt(
        tokenizer,
        chat,
        add_generation_prompt=chat.add_generation_prompt,
        continue_final_message=chat.continue_final_message,
    )
    id = generate_response_id()
    start_time = time.perf_counter()
    generated_text = ""

    response_stream = stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=make_sampler(**chat.sampler_params),
        max_tokens=chat.max_tokens,
    )

    first_response = next(response_stream, None)
    initial_usage = {
        "input_tokens": _resolve_prompt_tokens(first_response, prompt),
        "output_tokens": _resolve_generation_tokens(first_response),
    }

    yield MessageStartEvent(id, chat.request_model, initial_usage).emit()
    yield ContentBlockStartEvent().emit()

    response = first_response
    if first_response and first_response.text:
        generated_text += first_response.text
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[assistantΔ] %s", first_response.text)
        yield ContentBlockDeltaEvent("text_delta", first_response.text).emit()

    for response in response_stream:
        generated_text += response.text
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[assistantΔ] %s", response.text)
        yield ContentBlockDeltaEvent("text_delta", response.text).emit()

    prompt_tokens = _resolve_prompt_tokens(response, prompt)
    generation_tokens = _resolve_generation_tokens(response)
    stop_reason = _resolve_stop_reason(response)
    final_usage = {
        "input_tokens": prompt_tokens,
        "output_tokens": generation_tokens,
    }

    yield ContentBlockStopEvent().emit()
    yield MessageDeltaEvent(stop_reason, usage=final_usage).emit()
    yield MessageStopEvent().emit()

    _log_generation_summary(
        generated_text=generated_text,
        last_response=response,
        elapsed_seconds=time.perf_counter() - start_time,
        is_streaming=True,
    )


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
