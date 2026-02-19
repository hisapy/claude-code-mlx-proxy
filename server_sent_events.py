from abc import ABC
import logging
import json


logger = logging.getLogger("uvicorn.error")


class SSEvent(ABC):
    def emit(self):
        logger.debug(f"Emitting event: {self.type} with data: {self.data}")
        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.type}\ndata: {payload}\n\n"


class MessageStartEvent(SSEvent):
    def __init__(self, id: str, model: str, usage: dict[str, int]):
        self.type = "message_start"
        self.data = {
            "type": self.type,
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        }


class ContentBlockStartEvent(SSEvent):
    def __init__(self, content_block: dict, index=0):
        self.type = "content_block_start"
        self.data = {
            "type": self.type,
            "index": index,
            "content_block": content_block,
        }


class ContentBlockDeltaEvent(SSEvent):
    def __init__(self, type, value, index=0):
        self.type = "content_block_delta"

        delta = {"type": type}

        if type == "text_delta":
            delta["text"] = value
        elif type == "input_json_delta":
            delta["partial_json"] = value
        elif type == "thinking_delta":
            delta["thinking"] = value
        elif type == "signature_delta":
            delta["signature"] = value

        self.data = {
            "type": self.type,
            "index": index,
            "delta": delta,
        }


class ContentBlockStopEvent(SSEvent):
    def __init__(self, index=0):
        self.type = "content_block_stop"
        self.data = {"type": self.type, "index": index}


class MessageDeltaEvent(SSEvent):
    def __init__(self, stop_reason, stop_sequence=None, usage=None):
        self.type = "message_delta"

        self.data = {
            "type": self.type,
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": usage,
        }


class MessageStopEvent(SSEvent):
    def __init__(self):
        self.type = "message_stop"
        self.data = {"type": self.type}
