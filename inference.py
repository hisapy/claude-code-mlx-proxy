import logging
from typing import Dict, TypedDict, Literal, Optional
from pydantic import BaseModel


from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler

from schemas import ClaudeMessageParams
from config import settings

logger = logging.getLogger(__name__)


def load_llm(model_name: str, trust_remote_code: bool):
    logger.info(f"Loading MLX model: {model_name}")
    model, tokenizer = load(
        settings.model_name,
        tokenizer_config={"trust_remote_code": trust_remote_code},
    )
    logger.info("Model loaded successfully!")

    return model, tokenizer


class ToolFunctionParametersProperties(TypedDict):
    type: str
    description: str


class ToolFunctionParameters(TypedDict):
    type: str
    properties: Dict[str, ToolFunctionParametersProperties]
    required: list[str]


class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: ToolFunctionParameters


class ToolUse(TypedDict):
    type: Literal["function"]
    function: ToolFunction


class Message(TypedDict):
    role: str
    content: str


class ChatParams(BaseModel):
    """
    Internal chat representation
    """

    messages: list[Message]
    tools: list[ToolUse] = []
    max_tokens: int
    sampler: Optional[object] = None
    thinking_enabled: bool = False


def parse_claude_message_params(params: ClaudeMessageParams) -> ChatParams:
    return ChatParams(
        messages=_parse_conversation(params),
        max_tokens=4069,
        # tools=_parse_tools(params),
        # max_tokens=_parse_max_tokens(params),
        # sampler = make_sampler(params.temperature, params.top_p, params.top_k)
    )


def _parse_conversation(params: ClaudeMessageParams):
    messges = []
    system_prompt = params.system

    if isinstance(system_prompt, str):
        messges.append({"role": "system", "content": system_prompt})
    else:  # is a list
        for msg in system_prompt:
            messges.append({"role": "system", "content": msg.text})

    return messges


def _parse_tools(params: ClaudeMessageParams):
    if not params.tools:
        return None
    tools = []
    for tool in params.tools:
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
        )
    return tools


def _parse_max_tokens(params: ClaudeMessageParams):
    max_tokens = 4096
    return max_tokens


def claude_chat(model, tokenizer, params: ChatParams):
    prompt = build_prompt(tokenizer, params)

    return generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=params.sampler,
        max_tokens=params.max_tokens,
        verbose=settings.verbose,
    )


def claude_chat_stream(model, tokenizer, params: ChatParams):
    chat = parse_claude_message_params(params)
    prompt = build_prompt(chat)

    # yield MessageStartEvent()
    # yield ContentBlockStartEvent()

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        sampler=chat.sampler,
        max_tokens=chat.max_tokens,
        verbose=settings.verbose,
    ):
        # Include params.model in the reponse
        # yield ContentBlockDeltaEvent(response.text)
        print(response.text)
        yield response.text

    # yield ContentBlockStopEvent()
    # yield MessageDeltaEvent()
    # yield MessageStopEvent()


def build_prompt(tokenizer, chat: ChatParams):
    return tokenizer.apply_chat_template(
        chat.messages,
        tools=chat.tools,
        enable_thinking=chat.thinking_enabled,
        add_generation_prompt=True,  # TODO: what happens with a "PREFILL response" in the request?
        tokenize=True,  # True because not adding special tokens
    )


def claude_tokens_count(text: str) -> int:
    pass
    # """Count tokens in text"""
    # try:
    #     # MLX tokenizers often expect the text to be handled through their specific methods
    #     # First try the standard approach with proper string handling
    #     if isinstance(text, str) and text.strip():
    #         # For MLX, we may need to use a different approach
    #         # Try to get tokens using the tokenizer's __call__ method or encode
    #         try:
    #             # Some MLX tokenizers work better with this approach
    #             result = tokenizer(text, return_tensors=False, add_special_tokens=False)
    #             if isinstance(result, dict) and "input_ids" in result:
    #                 return len(result["input_ids"])
    #             elif hasattr(result, "__len__"):
    #                 return len(result)
    #         except (AttributeError, TypeError, ValueError):
    #             pass

    #         # Try direct encode without parameters
    #         try:
    #             encoded = tokenizer.encode(text)
    #             return (
    #                 len(encoded) if hasattr(encoded, "__len__") else len(list(encoded))
    #             )
    #         except (AttributeError, TypeError, ValueError):
    #             pass

    #         # Try with explicit string conversion and basic parameters
    #         try:
    #             tokens = tokenizer.encode(str(text), add_special_tokens=False)
    #             return len(tokens)
    #         except (AttributeError, TypeError, ValueError):
    #             pass

    #     # Final fallback: character-based estimation
    #     return max(1, len(str(text)) // 4)  # At least 1 token, ~4 chars per token

    # except Exception as e:
    #     print(f"Token counting failed with error: {e}")
    #     return max(1, len(str(text)) // 4)  # Fallback estimation


# async def stream_generate_response(
#     request: MessagesRequest, prompt: str, input_tokens: int
# ):
#     """Generate streaming response"""
#     response_id = "msg_" + str(abs(hash(prompt)))[:8]
#     full_text = ""

#     # Send message start event
#     message_start = {
#         "type": "message_start",
#         "message": {
#             "id": response_id,
#             "type": "message",
#             "role": "assistant",
#             "content": [],
#             "model": request.model,
#             "stop_reason": None,
#             "stop_sequence": None,
#             "usage": {"input_tokens": input_tokens, "output_tokens": 0},
#         },
#     }
#     yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

#     # Send content block start
#     content_start = {
#         "type": "content_block_start",
#         "index": 0,
#         "content_block": {"type": "text", "text": ""},
#     }
#     yield f"event: content_block_start\ndata: {json.dumps(content_start)}\n\n"

#     # Stream generation
#     for i, response in enumerate(
#         stream_generate(
#             model,
#             tokenizer,
#             prompt=prompt,
#             max_tokens=request.max_tokens,
#         )
#     ):
#         full_text += response.text

#         # Send content block delta
#         content_delta = {
#             "type": "content_block_delta",
#             "index": 0,
#             "delta": {"type": "text_delta", "text": response.text},
#         }
#         yield f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n"

#     # Count output tokens
#     output_tokens = count_tokens(full_text)

#     # Send content block stop
#     content_stop = {"type": "content_block_stop", "index": 0}
#     yield f"event: content_block_stop\ndata: {json.dumps(content_stop)}\n\n"

#     # Send message delta with usage
#     message_delta = {
#         "type": "message_delta",
#         "delta": {"stop_reason": "end_turn", "stop_sequence": None},
#         "usage": {"output_tokens": output_tokens},
#     }
#     yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

#     # Send message stop
#     message_stop = {"type": "message_stop"}
#     yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
