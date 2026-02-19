import uuid
import time
import logging
import importlib
import json
import shlex
from itertools import tee
from contextlib import contextmanager

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


async def claude_chat(model, tokenizer, chat_params: ClaudeMessageParams):
    chat = claude_parser.parse_chat_params(chat_params)
    stop_sequences = _resolve_stop_sequences(chat)
    generation_kwargs = _build_generation_kwargs(chat)
    # logger.debug(f"Parsed chat:\n%s", chat.model_dump_json(indent=2))

    with inference_profiler() as profiler:
        prompt = build_prompt(tokenizer, chat, tokenize=True)
        prompt_input, prompt_tokens = _prepare_prompt_input(tokenizer, prompt)
        profiler["effective_prompt_tokens"] = prompt_tokens

        response_generator = stream_generate(
            model,
            tokenizer,
            prompt=prompt_input,
            **generation_kwargs,
        )

        generated_text = ""
        response = None
        stream_sanitizer = claude_parser.create_stream_text_sanitizer()
        text_stopper = _TextStopper(stop_sequences)
        matched_stop_sequence = None
        for response in response_generator:
            profiler["last_response"] = response
            cleaned = stream_sanitizer.push(response.text)
            if not cleaned:
                continue

            emit_text, matched = text_stopper.push(cleaned)
            if emit_text:
                generated_text += emit_text
            if matched:
                matched_stop_sequence = matched
                break

        if matched_stop_sequence is None:
            tail = stream_sanitizer.finish()
            if tail:
                emit_text, matched = text_stopper.push(tail)
                if emit_text:
                    generated_text += emit_text
                if matched:
                    matched_stop_sequence = matched

            if matched_stop_sequence is None:
                generated_text += text_stopper.finish()

        generated_text = claude_parser.sanitize_response_text(generated_text)

        logger.debug(f"Full generated text:\n{generated_text}")

        # TODO: handle other types of content block
        return ClaudeMessage(
            id=generate_response_id(),
            content=[TextBlock(text=generated_text)],
            model=chat.request_model,
            stop_reason=_resolve_stop_reason(response, matched_stop_sequence),
            stop_sequence=matched_stop_sequence,
            usage=Usage(
                input_tokens=response.prompt_tokens if response else 0,
                output_tokens=response.generation_tokens if response else 0,
            ),
        )


async def claude_chat_stream(model, tokenizer, chat_params: ClaudeMessageParams):
    chat = claude_parser.parse_chat_params(chat_params)
    stop_sequences = _resolve_stop_sequences(chat)
    generation_kwargs = _build_generation_kwargs(chat)
    # logger.debug(f"Parsed chat:\n%s", chat.model_dump_json(indent=2))

    with inference_profiler() as profiler:
        prompt = build_prompt(tokenizer, chat, tokenize=True)
        prompt_input, prompt_tokens = _prepare_prompt_input(tokenizer, prompt)
        profiler["effective_prompt_tokens"] = prompt_tokens
        response_generator = stream_generate(
            model,
            tokenizer,
            prompt=prompt_input,
            **generation_kwargs,
        )
        available_tool_names = {
            tool.get("function", {}).get("name")
            for tool in (chat.tools or [])
            if isinstance(tool, dict)
        }
        available_tool_names.discard(None)

        segment_parser = claude_parser.create_stream_segment_parser(
            enable_thinking=chat.enable_thinking
        )
        text_stopper = _TextStopper(stop_sequences)
        current_block_type = None
        current_block_index = -1

        async def emit_segment(segment: dict):
            nonlocal current_block_type, current_block_index, matched_stop_sequence

            segment_kind = segment.get("kind")
            if segment_kind in {"text", "thinking"}:
                segment_text = (
                    segment.get("thinking", "")
                    if segment_kind == "thinking"
                    else segment.get("text", "")
                )
                emit_text, matched = text_stopper.push(segment_text)
                if matched:
                    matched_stop_sequence = matched
                if not emit_text:
                    return

                target_block_type = "thinking" if segment_kind == "thinking" else "text"
                delta_type = "thinking_delta" if segment_kind == "thinking" else "text_delta"

                if current_block_type != target_block_type:
                    if current_block_type is not None:
                        yield ContentBlockStopEvent(index=current_block_index).emit()
                    current_block_index += 1
                    current_block_type = target_block_type
                    content_block = (
                        {"type": "thinking", "thinking": ""}
                        if target_block_type == "thinking"
                        else {"type": "text", "text": ""}
                    )
                    yield ContentBlockStartEvent(content_block, index=current_block_index).emit()

                yield ContentBlockDeltaEvent(delta_type, emit_text, index=current_block_index).emit()
                return

            if segment_kind == "tool_use":
                tool_name = segment.get("name")
                partial_json = segment.get("partial_json", "")

                tool_name, partial_json = _remap_unknown_tool_call(
                    tool_name=tool_name,
                    partial_json=partial_json,
                    available_tool_names=available_tool_names,
                )

                if not tool_name:
                    return

                if current_block_type is not None:
                    yield ContentBlockStopEvent(index=current_block_index).emit()
                    current_block_type = None

                current_block_index += 1
                yield ContentBlockStartEvent(
                    {
                        "type": "tool_use",
                        "id": generate_response_id(prefix="toolu"),
                        "name": tool_name,
                        "input": {},
                    },
                    index=current_block_index,
                ).emit()
                yield ContentBlockDeltaEvent("input_json_delta", partial_json, index=current_block_index).emit()
                yield ContentBlockStopEvent(index=current_block_index).emit()

        # We need to "peek" the first item to get the first usage
        it1, it2 = tee(response_generator)
        first_response = next(it1, None)
        if first_response is not None:
            profiler["last_response"] = first_response

        usage = {
            "input_tokens": first_response.prompt_tokens if first_response else 0,
            "output_tokens": first_response.generation_tokens if first_response else 0,
        }

        id = generate_response_id()
        yield MessageStartEvent(id, model=chat.request_model, usage=usage).emit()

        if first_response is None:
            yield MessageDeltaEvent("end_turn", usage=usage).emit()
            yield MessageStopEvent().emit()
            return

        matched_stop_sequence = None
        for response in it2:
            profiler["last_response"] = response
            for segment in segment_parser.push(response.text):
                async for event in emit_segment(segment):
                    yield event
            if matched_stop_sequence:
                break

        for segment in segment_parser.finish():
            async for event in emit_segment(segment):
                yield event

        if matched_stop_sequence is None:
            tail = text_stopper.finish()
            if tail:
                async for event in emit_segment({"kind": "text", "text": tail}):
                    yield event

        if current_block_type is not None:
            yield ContentBlockStopEvent(index=current_block_index).emit()

        usage = {
            "input_tokens": response.prompt_tokens,
            "output_tokens": response.generation_tokens,
        }
        yield MessageDeltaEvent(
            _resolve_stop_reason(response, matched_stop_sequence),
            stop_sequence=matched_stop_sequence,
            usage=usage,
        ).emit()
        yield MessageStopEvent().emit()

    logger.debug(f"### Stream completed ###")


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
    chat.add_generation_prompt = False
    chat.continue_final_message = False

    tokens = build_prompt(tokenizer, chat, tokenize=True)

    return ClaudeTokenCount(input_tokens=len(tokens))


def build_prompt(tokenizer, chat: ChatParams, tokenize: bool = False):
    prompt = _render_prompt(tokenizer, chat, tokenize=tokenize)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("### chat:\n%s", chat.model_dump_json(indent=2))
        if tokenize:
            logger.debug(
                "### prompt:\n%s", _render_prompt(tokenizer, chat, tokenize=False)
            )
        else:
            logger.debug("### prompt:\n%s", prompt)

    return prompt


def _render_prompt(tokenizer, chat: ChatParams, tokenize: bool):
    return tokenizer.apply_chat_template(
        chat.messages,
        tools=chat.tools,
        enable_thinking=chat.enable_thinking,
        add_generation_prompt=chat.add_generation_prompt,
        continue_final_message=chat.continue_final_message,
        tokenize=tokenize,
    )


def generate_response_id(prefix="msg"):
    return f"{prefix}_{uuid.uuid4().hex[:34]}"


@contextmanager
def inference_profiler():
    start_time = time.perf_counter()
    stats = {"last_response": None}
    try:
        yield stats

    finally:
        elapsed_seconds = time.perf_counter() - start_time
        last_response = stats.get("last_response") if isinstance(stats, dict) else None
        if last_response is None:
            logger.info(f"Inference took {elapsed_seconds:.3f} seconds")
            return

        logger.info(
            "Inference took %.3f seconds | prompt_tokens=%s effective_prompt_tokens=%s generation_tokens=%s prompt_tps=%s generation_tps=%s peak_memory_gb=%s finish_reason=%s",
            elapsed_seconds,
            getattr(last_response, "prompt_tokens", None),
            stats.get("effective_prompt_tokens") if isinstance(stats, dict) else None,
            getattr(last_response, "generation_tokens", None),
            getattr(last_response, "prompt_tps", None),
            getattr(last_response, "generation_tps", None),
            getattr(last_response, "peak_memory", None),
            getattr(last_response, "finish_reason", None),
        )


def _prepare_prompt_input(tokenizer, prompt_input):
    prompt_tokens = None

    if isinstance(prompt_input, str):
        if not settings.max_input_tokens or settings.max_input_tokens <= 0:
            return prompt_input, None
        try:
            prompt_tokens = tokenizer.encode(prompt_input)
        except Exception:
            logger.debug(
                "Falling back to raw prompt string; tokenizer.encode unavailable"
            )
            return prompt_input, None
    else:
        prompt_tokens = prompt_input

    original_count = len(prompt_tokens)
    if not settings.max_input_tokens or settings.max_input_tokens <= 0:
        return prompt_tokens, original_count

    if original_count <= settings.max_input_tokens:
        return prompt_tokens, original_count

    trimmed_tokens = prompt_tokens[-settings.max_input_tokens :]
    logger.debug(
        "Trimmed prompt tokens from %s to %s (MAX_INPUT_TOKENS)",
        original_count,
        settings.max_input_tokens,
    )
    return trimmed_tokens, len(trimmed_tokens)


def _resolve_stop_reason(response, matched_stop_sequence: str | None) -> str:
    """Map MLX finish_reason to Claude stop_reason."""
    if matched_stop_sequence:
        return "end_turn"
    elif response.finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _resolve_stop_sequences(chat: ChatParams) -> list[str]:
    sequences = []
    seen = set()

    for value in chat.stop_sequences or []:
        if not value or value in seen:
            continue
        seen.add(value)
        sequences.append(value)

    for value in claude_parser.default_stop_sequences():
        if not value or value in seen:
            continue
        seen.add(value)
        sequences.append(value)

    if settings.eos_token and settings.eos_token not in seen:
        sequences.append(settings.eos_token)

    return sequences


def _build_generation_kwargs(chat: ChatParams) -> dict:
    max_tokens = _resolve_max_tokens(chat.max_tokens)
    kwargs = {
        "sampler": make_sampler(**chat.sampler_params),
        "max_tokens": max_tokens,
    }

    if settings.max_kv_size and settings.max_kv_size > 0:
        kwargs["max_kv_size"] = settings.max_kv_size

    if max_tokens < chat.max_tokens:
        logger.debug(
            "Clamped max_tokens from %s to %s (DEFAULT_MAX_TOKENS)",
            chat.max_tokens,
            max_tokens,
        )

    return kwargs


def _resolve_max_tokens(requested: int) -> int:
    if settings.default_max_tokens and settings.default_max_tokens > 0:
        return max(1, min(requested, settings.default_max_tokens))
    return max(1, requested)


def _remap_unknown_tool_call(
    tool_name: str | None,
    partial_json: str,
    available_tool_names: set[str],
) -> tuple[str | None, str]:
    if not tool_name:
        return tool_name, partial_json

    if tool_name in available_tool_names:
        return tool_name, partial_json

    alias_map = {
        "ListFiles": "Bash",
        "ListDirectory": "Bash",
        "ls": "Bash",
        "ReadFile": "Read",
    }

    mapped_tool = alias_map.get(tool_name)
    if not mapped_tool or mapped_tool not in available_tool_names:
        return tool_name, partial_json

    logger.debug(
        "Remapping unsupported tool call '%s' -> '%s'",
        tool_name,
        mapped_tool,
    )

    parsed_input = {}
    if partial_json:
        try:
            maybe_dict = json.loads(partial_json)
            if isinstance(maybe_dict, dict):
                parsed_input = maybe_dict
        except Exception:
            parsed_input = {}

    if mapped_tool == "Bash":
        directory = parsed_input.get("directory") or parsed_input.get("path") or "."
        if not isinstance(directory, str) or not directory.strip():
            directory = "."

        command = f"ls -la -- {shlex.quote(directory)}"
        mapped_input = {
            "command": command,
            "description": "List files in directory",
        }
        return mapped_tool, json.dumps(mapped_input, ensure_ascii=False)

    if mapped_tool == "Read":
        file_path = (
            parsed_input.get("file_path")
            or parsed_input.get("path")
            or parsed_input.get("file")
        )
        if isinstance(file_path, str) and file_path:
            return mapped_tool, json.dumps({"file_path": file_path}, ensure_ascii=False)

    return tool_name, partial_json


class _TextStopper:
    def __init__(self, stop_sequences: list[str]):
        self.stop_sequences = [sequence for sequence in stop_sequences if sequence]
        self.trailing_window_size = (
            max(len(sequence) for sequence in self.stop_sequences) - 1
            if self.stop_sequences
            else 0
        )
        self.buffer = ""
        self.stopped = False

    def push(self, text: str) -> tuple[str, str | None]:
        if self.stopped or not text:
            return "", None

        if not self.stop_sequences:
            return text, None

        self.buffer += text
        truncated, matched = _apply_stop_sequences(self.buffer, self.stop_sequences)
        if matched:
            self.stopped = True
            self.buffer = ""
            return truncated, matched

        if len(self.buffer) <= self.trailing_window_size:
            return "", None

        emit = self.buffer[: -self.trailing_window_size]
        self.buffer = self.buffer[-self.trailing_window_size :]
        return emit, None

    def finish(self) -> str:
        if self.stopped:
            return ""
        tail = self.buffer
        self.buffer = ""
        return tail


def _apply_stop_sequences(
    text: str, stop_sequences: list[str]
) -> tuple[str, str | None]:
    if not text or not stop_sequences:
        return text, None

    earliest_index = None
    matched_stop_sequence = None

    for sequence in stop_sequences:
        index = text.find(sequence)
        if index == -1:
            continue

        if earliest_index is None or index < earliest_index:
            earliest_index = index
            matched_stop_sequence = sequence

    if matched_stop_sequence is None:
        return text, None

    return text[:earliest_index], matched_stop_sequence
