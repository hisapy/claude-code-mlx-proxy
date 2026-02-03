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

    # Generation settings
    default_max_tokens: int = 4096
    default_temperature: float = 1.0
    default_top_p: float = 1.0

    # Verbosity
    verbose: bool = False


settings = Settings()
