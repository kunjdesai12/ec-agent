"""Centralized configuration loaded from environment variables."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # vLLM upstream (OpenAI-compatible)
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"  # vLLM ignores but openai client requires
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    vllm_request_timeout_s: float = 60.0

    # Generation
    max_new_tokens: int = 400
    temperature: float = 0.3
    top_p: float = 0.9

    # Valkey/Redis
    redis_url: str = "redis://valkey:6379/0"
    history_ttl_s: int = 60 * 60 * 6  # 6h
    history_max_turns: int = 12

    # RAG
    embedding_model: str = "BAAI/bge-m3"
    rag_top_k: int = 5

    # Tool loop
    max_tool_iterations: int = 5

    # Semantic search url
    semantic_search_url: str = "http://localhost:8000"

    # pgvector (RDS Postgres — vector store)
    pg_host: str = "easycater-vector-db-v2.cngqs42w0l3u.ap-south-1.rds.amazonaws.com"
    pg_port: int = 5432
    pg_db: str = "easycater_vectors"
    pg_user: str = "vectoradmin"
    pg_password: str = "vector123"  # set via PG_PASSWORD env var or .env
    use_memory_store: bool = True
    tool_base_url: str = "http://localhost:3002"

@lru_cache
def get_settings() -> Settings:
    return Settings()