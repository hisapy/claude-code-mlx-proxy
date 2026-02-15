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
    claude_tokens_count,
)


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
    return claude_tokens_count(tokenizer, params)


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
