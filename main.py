import logging
import uvicorn

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from config import settings
from claude_schemas import (
    ClaudeMessageParams,
    ClaudeTokenCountParams,
    ClaudeMessage,
    ClaudeTokenCount,
)


from inference import (
    claude_parser,
    load_llm,
    claude_chat,
    claude_chat_stream,
    # claude_tokens_count,
)


# NOTICE: Actually, thgst code is not just a proxy, it also loads/starts the LLM.
# If multiple workers are started (e.g., --workers 4), each process will load its own model
# Maybe this should start or reference a mlx_lm.server which can be scaled separately

# MLX is based on Transformers and transformers is the model definition framework
# ... so we need to convert Claude Code requests (Claude API) to MLX/Transformers format
# - The system prompt goes to the beginning of the messages
# - The tools are passed as a kwarg to apply_chat_template
# - In the tools we need to convert input_schema to parameters
# - It seems thinking can be enabled/disabled with the enable_thinking kwarg in apply_chat_template
# - ERROR if the model doesn't have a chat template (currently it fallbacks to manual formatting)

# Where do we pass temperature, top_p, top_k, etc?
#      (we can pass the sample kwarg to the generate function)


# If prompted to trust_remote_code, then where do we pass it? generate or apply_chat_template?
# - We should not worry about eos_token when loading the model
#   See https://huggingface.co/docs/transformers/en/chat_templating where it talks about special <bos> and <eos>

# Logger level should be used to control logging

# FOR THE COMMIT MESSAGE

# Use uvicorn logger properly
# Separate the inference layer from the web layer
# Omit the eos_token, see the text in red in https://huggingface.co/docs/transformers/en/chat_templating#using-applychattemplate
# Tokenize the chat because we are not adding special tokens manually
# Support thinking
# Fix tools support
# Calculate max_tokens properly, substracting tokens from thinking
# Rename models for similarity with Claude API docs
# Support sampler (temperature, top_p, top_k)
# Use chat template from config (NOTICE that not all chat templates support list content or tools, e.g., chat_template.jinja from qwen3)


# TODO with Claude Code:
# - Generate Claude.md
# - Tell to document (or mix task equivalent) the device_info.py script
# - Maybe add device_info to OpenAPI docs page (or just a json endpoint???)
# - Basic CI/CD???

# Global variables for model and tokenizer
model = None
tokenizer = None

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, tokenizer
    model, tokenizer = load_llm(settings.model_name, settings.trust_remote_code)
    yield
    logger.info("Shutting down...")


async def verify_model_loaded():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


app = FastAPI(lifespan=lifespan, dependencies=[Depends(verify_model_loaded)])


# TODO: document streaming response
@app.post("/v1/messages")
async def create_message(params: ClaudeMessageParams) -> ClaudeMessage:
    chat = claude_parser.parse_chat_params(params)

    if params.stream:
        return StreamingResponse(
            claude_chat_stream(model, tokenizer, chat),
            media_type="text/event-stream",
        )
    else:
        return await claude_chat(model, tokenizer, chat)


@app.post("/v1/messages/count_tokens")
async def count_tokens(params: ClaudeTokenCountParams) -> ClaudeTokenCount:
    logger.debug("*** count_tokens called ***")
    logger.debug(f"last message: {params.messages[-1]}")
    # return claude_tokens_count(params)
    return ClaudeTokenCount(input_tokens=100)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
async def root():
    return {
        "message": "Claude Code MLX Proxy",
        "status": "running",
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    print(f"Starting Claude Code MLX Proxy on {settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port)
