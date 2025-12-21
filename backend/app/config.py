"""Application configuration using environment variables."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # API settings
    api_title: str = "Trading Application API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api"
    
    # Logging
    log_level: str = "INFO"
    
    # Database settings
    database_url: str = "sqlite+aiosqlite:///./trading.db"
    
    # MT5 settings
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_enabled: bool = True  # Set to False to force mock mode
    mt5_real_trading: bool = True  # Set to True to enable real MT5 trades (requires proper setup)
    
    # Data scraping settings
    alpha_vantage_api_key: Optional[str] = None
    min_rows_per_symbol: int = 10000  # Minimum rows to scrape per symbol
    max_rows_per_symbol: int = 10000  # Maximum rows per symbol (rolling window) - 10,000 limit
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
