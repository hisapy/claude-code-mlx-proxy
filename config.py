from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the Claude Code MLX Proxy"""

    # Read ENV vars from an .env file
    model_config = SettingsConfigDict(env_file=".env")

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8888

    # Model settings
    model_name: str = "mlx-community/GLM-4.5-Air-3bit"
    trust_remote_code: bool = False
    eos_token: str | None = None
    # The default chat_template came from:
    # https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/chat_template.jinja

    # Generation settings
    claude_mlx_adapter: str
    default_max_tokens: int = 4096
    default_temperature: float = 1.0

    # Verbosity
    verbose: bool = False


settings = Settings()
