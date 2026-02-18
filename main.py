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
    # _debug_incoming_message(params)

    if params.stream:
        return StreamingResponse(
            claude_chat_stream(model, tokenizer, params),
            media_type="text/event-stream",
        )
    else:
        return await claude_chat(model, tokenizer, params)


@app.post("/v1/messages/count_tokens")
async def count_tokens(params: ClaudeTokenCountParams) -> ClaudeTokenCount:
    # _debug_incoming_message(params)
    return claude_tokens_count(tokenizer, params)


def _debug_incoming_message(params):
    logger.debug(
        f"Claude Code request\n stream: %s\n has_tools: %s\n thinking: %s\nlast message:\n%s",
        getattr(params, "stream", False),
        len(params.tools) > 0 if params.tools else False,
        params.thinking,
        params.messages[-1].model_dump_json(indent=2),
    )


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
