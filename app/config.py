from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App settings
    app_name: str = "Aegis AI Backend"
    debug: bool = False

    # Security
    secret_key: str

    # Database
    database_url: str

    # Google OAuth
    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str

    # Frontend
    frontend_url: str

    # Backend URL (for file uploads)
    backend_url: str = ""

    # AI Gateway (for LLM API calls)
    ai_gateway_api_key: str = ""

    # LLM API Keys (provider-specific)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    xai_api_key: str = ""

    # Rate Limiting
    guest_message_limit: int = 20
    regular_user_message_limit: int = 50

    # File Upload Settings
    max_file_size_mb: int = 5
    allowed_image_types: list[str] = ["image/jpeg", "image/png"]

    # Model Configuration
    # Using Claude models (Anthropic) - 2025 models
    default_chat_model: str = "claude/claude-haiku-4-5-20251001"
    default_title_model: str = "claude/claude-haiku-4-5-20251001"
    default_artifact_model: str = "claude/claude-haiku-4-5-20251001"

    # Agent Settings
    agent_enabled: bool = True  # Enable/disable LangGraph agent
    agent_model: str = "claude-haiku-4-5-20251001"  # Model for agent reasoning (Haiku 4.5 is fast and cost-effective)
    max_tool_steps: int = 5  # Maximum number of tool execution iterations
    thinking_budget_tokens: int = 10000  # Token budget for extended thinking

    # Redis (optional, for future use)
    redis_url: str = ""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
