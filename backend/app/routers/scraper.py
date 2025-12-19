"""Data scraping endpoints."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from app.services.scraper import scraper_service
from app.utils.logging import get_logger
from app.database import db
from app.models import OHLCV
from sqlalchemy import select, func

logger = get_logger(__name__)
router = APIRouter(prefix="/api/scrape", tags=["scraping"])


class ScrapeRequest(BaseModel):
    """Request model for data scraping."""
    symbol: str = Field(..., description="Trading symbol to scrape")
    sources: List[str] = Field(..., description="List of sources: yahoo, alphavantage, ccxt, custom")
    source_params: Optional[Dict[str, Dict]] = Field(None, description="Optional parameters for each source")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "sources": ["yahoo", "alphavantage"],
                "source_params": {
                    "yahoo": {"period": "max"},
                    "alphavantage": {"interval": "daily"}
                }
            }
        }


class CustomSourceRequest(BaseModel):
    """Request model for custom source scraping."""
    symbol: str = Field(..., description="Trading symbol")
    url: str = Field(..., description="URL to fetch data from")
    column_mapping: Dict[str, str] = Field(
        ...,
        description="Column mapping: {timestamp: 'date_col', open: 'open_col', ...}",
        alias="schema"  # Accept 'schema' in JSON to avoid shadowing warning
    )
    
    class Config:
        populate_by_name = True  # Allow both 'schema' and 'column_mapping'


class ScrapeResponse(BaseModel):
    """Response model for scraping operations."""
    progress_key: str
    symbol: str
    status: str
    message: str


@router.post("/start", response_model=ScrapeResponse)
async def start_scraping(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Start scraping historical data from multiple sources.
    
    Scrapes data concurrently from all specified sources and stores it in the database.
    Each source runs independently - if one fails, others continue.
    
    Minimum target: 10,000 rows per symbol (configurable).
    """
    try:
        # Generate progress key
        progress_key = f"{request.symbol}_{datetime.utcnow().isoformat()}"
        
        # Start scraping in background
        background_tasks.add_task(
            scraper_service.scrape_multiple_sources,
            request.symbol,
            request.sources,
            request.source_params
        )
        
        return ScrapeResponse(
            progress_key=progress_key,
            symbol=request.symbol,
            status="started",
            message=f"Scraping started for {request.symbol} from {', '.join(request.sources)}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start scraping: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(e)}")


@router.post("/custom", response_model=ScrapeResponse)
async def scrape_custom_source(request: CustomSourceRequest, background_tasks: BackgroundTasks):
    """
    Scrape data from a custom URL source.
    
    Requires a schema mapping to map source columns to OHLCV format:
    - timestamp: Column name for timestamp
    - open: Column name for open price
    - high: Column name for high price
    - low: Column name for low price
    - close: Column name for close price
    - volume: Column name for volume
    """
    try:
        progress_key = f"{request.symbol}_custom_{datetime.utcnow().isoformat()}"
        
        # Start scraping in background
        background_tasks.add_task(
            scraper_service.scrape_custom,
            request.symbol,
            request.url,
            request.column_mapping,
            progress_key
        )
        
        return ScrapeResponse(
            progress_key=progress_key,
            symbol=request.symbol,
            status="started",
            message=f"Custom source scraping started for {request.symbol}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start custom scraping: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(e)}")


@router.get("/progress/{progress_key}")
async def get_scraping_progress(progress_key: str):
    """
    Get scraping progress for a given progress key.
    
    Returns current status, rows scraped per source, and total rows.
    """
    progress = scraper_service.get_progress(progress_key)
    
    if not progress:
        raise HTTPException(status_code=404, detail="Progress key not found")
    
    # Get current row count from database
    async for session in db.get_session():
        count_query = select(func.count(OHLCV.id)).where(OHLCV.symbol == progress.get('symbol', ''))
        result = await session.execute(count_query)
        total_rows = result.scalar() or 0
        break
    
    progress['current_total_rows'] = total_rows
    
    return progress


@router.get("/status/{symbol}")
async def get_symbol_status(symbol: str):
    """
    Get scraping status for a symbol.
    
    Returns total rows in database, rows per source, and whether minimum threshold is met.
    """
    async for session in db.get_session():
        # Total rows
        total_query = select(func.count(OHLCV.id)).where(OHLCV.symbol == symbol)
        total_result = await session.execute(total_query)
        total_rows = total_result.scalar() or 0
        
        # Rows per source
        source_query = select(
            OHLCV.source,
            func.count(OHLCV.id).label('count')
        ).where(
            OHLCV.symbol == symbol
        ).group_by(OHLCV.source)
        
        source_result = await session.execute(source_query)
        rows_per_source = {row.source or 'unknown': row.count for row in source_result}
        
        # Check minimum threshold
        from app.config import settings
        meets_minimum = total_rows >= settings.min_rows_per_symbol
        
        break
    
    return {
        'symbol': symbol,
        'total_rows': total_rows,
        'rows_per_source': rows_per_source,
        'meets_minimum': meets_minimum,
        'minimum_required': settings.min_rows_per_symbol
    }


@router.get("/sources")
async def list_available_sources():
    """
    List available data sources and their status.
    """
    sources = {
        'yahoo': {
            'name': 'Yahoo Finance',
            'available': True,
            'requires_auth': False,
            'description': 'Free stock and forex data via yfinance',
            'supported_symbols': 'Stocks, ETFs, Forex pairs (e.g., AAPL, EURUSD=X)'
        },
        'alphavantage': {
            'name': 'Alpha Vantage',
            'available': True,
            'requires_auth': True,
            'api_key_configured': bool(scraper_service.__class__.__module__ and hasattr(scraper_service, '_check_alpha_vantage')),
            'description': 'Premium market data API',
            'supported_symbols': 'Stocks, Forex, Crypto'
        },
        'ccxt': {
            'name': 'CCXT (Crypto Exchanges)',
            'available': True,
            'requires_auth': False,
            'description': 'Cryptocurrency exchange data',
            'supported_symbols': 'Crypto pairs (e.g., BTC/USDT, ETH/USDT)',
            'default_exchange': 'binance'
        },
        'custom': {
            'name': 'Custom Source',
            'available': True,
            'requires_auth': False,
            'description': 'User-provided URL with schema mapping',
            'supported_formats': 'JSON, CSV'
        }
    }
    
    # Check Alpha Vantage API key
    from app.config import settings
    sources['alphavantage']['api_key_configured'] = bool(settings.alpha_vantage_api_key)
    
    return {
        'sources': sources,
        'minimum_rows_per_symbol': settings.min_rows_per_symbol
    }
