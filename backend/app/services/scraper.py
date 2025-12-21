"""Historical data scraping service with multiple sources."""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import asyncio
from app.utils.logging import get_logger
from app.config import settings
from app.database import db
from app.models import OHLCV
from app.services.integrity import integrity_service
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

logger = get_logger(__name__)


class ScraperService:
    """Service for scraping historical OHLCV data from multiple sources."""
    
    def __init__(self):
        self._progress: Dict[str, Dict] = {}
    
    async def scrape_yahoo_finance(
        self, 
        symbol: str, 
        period: str = "max",
        progress_key: Optional[str] = None
    ) -> Tuple[int, Optional[str]]:
        """
        Scrape data from Yahoo Finance using yfinance.
        
        Args:
            symbol: Trading symbol (e.g., AAPL, EURUSD=X)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            progress_key: Optional key for progress tracking
            
        Returns:
            Tuple of (rows_scraped, error_message)
        """
        try:
            import yfinance as yf
            
            if progress_key:
                self._update_progress(progress_key, "yahoo", "starting", 0)
            
            logger.info(f"Scraping {symbol} from Yahoo Finance (period: {period})")
            
            # Download data - use 'max' period to get all available data including newest
            ticker = yf.Ticker(symbol)
            # Ensure we get the newest available data by using 'max' period
            effective_period = period if period != "max" else "max"
            df = ticker.history(period=effective_period)
            
            if df.empty:
                error = f"No data found for {symbol} on Yahoo Finance"
                logger.warning(error)
                if progress_key:
                    self._update_progress(progress_key, "yahoo", "failed", 0, error)
                return 0, error
            
            # Normalize timestamps to UTC and sort by timestamp (newest last)
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()  # Ensure chronological order
            
            # Verify data is recent (most recent timestamp should be within last 7 days)
            if not df.empty:
                most_recent = df.index.max()
                now = pd.Timestamp.now(tz='UTC')
                days_old = (now - most_recent).days
                
                if days_old > 7:
                    error = f"Yahoo Finance data is stale: most recent data is {days_old} days old (threshold: 7 days)"
                    logger.warning(error)
                    if progress_key:
                        self._update_progress(progress_key, "yahoo", "failed", 0, error)
                    return 0, error
                
                logger.info(f"Yahoo Finance data is recent: most recent timestamp is {most_recent} ({days_old} days old)")
            
            # Limit rows to max_rows_per_symbol before processing
            max_rows = settings.max_rows_per_symbol
            original_count = len(df)
            if len(df) > max_rows:
                # Keep most recent rows (tail gets the newest)
                df = df.tail(max_rows)
                logger.info(f"Limited Yahoo Finance data to {max_rows} most recent rows (was {original_count})")
            
            # Prepare data
            rows_to_insert = []
            async for session in db.get_session():
                # Check existing data
                existing_query = select(OHLCV).where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.source == "yahoo"
                    )
                )
                existing_result = await session.execute(existing_query)
                existing = existing_result.scalars().all()
                existing_timestamps = {row.timestamp for row in existing}
                
                # Prepare new rows
                for idx, row in df.iterrows():
                    if idx not in existing_timestamps:
                        rows_to_insert.append({
                            'symbol': symbol,
                            'timestamp': idx,
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': float(row['Volume']),
                            'source': 'yahoo'
                        })
                
                # Insert in batches
                if rows_to_insert:
                    batch_size = 1000
                    total_inserted = 0
                    for i in range(0, len(rows_to_insert), batch_size):
                        batch = rows_to_insert[i:i + batch_size]
                        ohlcv_objects = [OHLCV(**row) for row in batch]
                        session.add_all(ohlcv_objects)
                        await session.commit()
                        total_inserted += len(batch)
                        
                        if progress_key:
                            self._update_progress(
                                progress_key, 
                                "yahoo", 
                                "scraping", 
                                total_inserted,
                                f"Inserted {total_inserted} rows"
                            )
                    
                    # Enforce rolling window after all inserts
                    await integrity_service.enforce_rolling_window(symbol)
                    # Update metadata (will create new session)
                    await integrity_service.update_metadata(symbol)
                
                break  # Exit session context
            
            logger.info(f"Yahoo Finance: Scraped {len(rows_to_insert)} rows for {symbol}")
            if progress_key:
                self._update_progress(progress_key, "yahoo", "completed", len(rows_to_insert))
            
            return len(rows_to_insert), None
            
        except ImportError:
            error = "yfinance not installed. Install with: pip install yfinance"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "yahoo", "failed", 0, error)
            return 0, error
        except Exception as e:
            error = f"Yahoo Finance scraping failed: {str(e)}"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "yahoo", "failed", 0, error)
            return 0, error
    
    async def scrape_alpha_vantage(
        self,
        symbol: str,
        interval: str = "daily",
        progress_key: Optional[str] = None
    ) -> Tuple[int, Optional[str]]:
        """
        Scrape data from Alpha Vantage API.
        
        Args:
            symbol: Trading symbol
            interval: Data interval (daily, weekly, monthly)
            progress_key: Optional key for progress tracking
            
        Returns:
            Tuple of (rows_scraped, error_message)
        """
        try:
            from alpha_vantage.timeseries import TimeSeries
            
            if not settings.alpha_vantage_api_key:
                error = "Alpha Vantage API key not configured"
                logger.warning(error)
                if progress_key:
                    self._update_progress(progress_key, "alphavantage", "failed", 0, error)
                return 0, error
            
            if progress_key:
                self._update_progress(progress_key, "alphavantage", "starting", 0)
            
            logger.info(f"Scraping {symbol} from Alpha Vantage (interval: {interval})")
            
            # Initialize Alpha Vantage
            ts = TimeSeries(key=settings.alpha_vantage_api_key, output_format='pandas')
            
            # Get data based on interval - use 'full' to get all available data including newest
            if interval == "daily":
                data, _ = ts.get_daily(symbol=symbol, outputsize='full')  # 'full' gets all available data
            elif interval == "weekly":
                data, _ = ts.get_weekly(symbol=symbol)  # Weekly/monthly don't have outputsize param
            elif interval == "monthly":
                data, _ = ts.get_monthly(symbol=symbol)
            else:
                error = f"Unsupported interval: {interval}"
                if progress_key:
                    self._update_progress(progress_key, "alphavantage", "failed", 0, error)
                return 0, error
            
            if data.empty:
                error = f"No data found for {symbol} on Alpha Vantage"
                logger.warning(error)
                if progress_key:
                    self._update_progress(progress_key, "alphavantage", "failed", 0, error)
                return 0, error
            
            # Normalize timestamps to UTC and sort by timestamp (newest last)
            data.index = pd.to_datetime(data.index, utc=True)
            data = data.sort_index()  # Ensure chronological order
            
            # Verify data is recent (most recent timestamp should be within last 7 days)
            if not data.empty:
                most_recent = data.index.max()
                now = pd.Timestamp.now(tz='UTC')
                days_old = (now - most_recent).days
                
                if days_old > 7:
                    error = f"Alpha Vantage data is stale: most recent data is {days_old} days old (threshold: 7 days)"
                    logger.warning(error)
                    if progress_key:
                        self._update_progress(progress_key, "alphavantage", "failed", 0, error)
                    return 0, error
                
                logger.info(f"Alpha Vantage data is recent: most recent timestamp is {most_recent} ({days_old} days old)")
            
            # Limit rows to max_rows_per_symbol before processing
            max_rows = settings.max_rows_per_symbol
            if len(data) > max_rows:
                # Keep most recent rows
                data = data.tail(max_rows)
                logger.info(f"Limited Alpha Vantage data to {max_rows} most recent rows")
            
            # Prepare data
            rows_to_insert = []
            async for session in db.get_session():
                # Check existing data
                existing_query = select(OHLCV).where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.source == "alphavantage"
                    )
                )
                existing_result = await session.execute(existing_query)
                existing = existing_result.scalars().all()
                existing_timestamps = {row.timestamp for row in existing}
                
                # Prepare new rows
                for idx, row in data.iterrows():
                    if idx not in existing_timestamps:
                        rows_to_insert.append({
                            'symbol': symbol,
                            'timestamp': idx,
                            'open': float(row['1. open']),
                            'high': float(row['2. high']),
                            'low': float(row['3. low']),
                            'close': float(row['4. close']),
                            'volume': float(row['5. volume']),
                            'source': 'alphavantage'
                        })
                
                # Insert in batches
                if rows_to_insert:
                    batch_size = 1000
                    total_inserted = 0
                    for i in range(0, len(rows_to_insert), batch_size):
                        batch = rows_to_insert[i:i + batch_size]
                        ohlcv_objects = [OHLCV(**row) for row in batch]
                        session.add_all(ohlcv_objects)
                        await session.commit()
                        total_inserted += len(batch)
                        
                        if progress_key:
                            self._update_progress(
                                progress_key,
                                "alphavantage",
                                "scraping",
                                total_inserted,
                                f"Inserted {total_inserted} rows"
                            )
                    
                    # Enforce rolling window after all inserts
                    await integrity_service.enforce_rolling_window(symbol)
                    # Update metadata (will create new session)
                    await integrity_service.update_metadata(symbol)
                
                break  # Exit session context
            
            logger.info(f"Alpha Vantage: Scraped {len(rows_to_insert)} rows for {symbol}")
            if progress_key:
                self._update_progress(progress_key, "alphavantage", "completed", len(rows_to_insert))
            
            return len(rows_to_insert), None
            
        except ImportError:
            error = "alpha-vantage not installed. Install with: pip install alpha-vantage"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "alphavantage", "failed", 0, error)
            return 0, error
        except Exception as e:
            error = f"Alpha Vantage scraping failed: {str(e)}"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "alphavantage", "failed", 0, error)
            return 0, error
    
    async def scrape_ccxt(
        self,
        symbol: str,
        exchange: str = "binance",
        days: int = 365,
        progress_key: Optional[str] = None
    ) -> Tuple[int, Optional[str]]:
        """
        Scrape crypto data using CCXT.
        
        Args:
            symbol: Trading symbol (e.g., BTC/USDT)
            exchange: Exchange name (default: binance)
            days: Number of days of history to fetch
            progress_key: Optional key for progress tracking
            
        Returns:
            Tuple of (rows_scraped, error_message)
        """
        try:
            import ccxt
            
            if progress_key:
                self._update_progress(progress_key, "ccxt", "starting", 0)
            
            logger.info(f"Scraping {symbol} from {exchange} via CCXT (days: {days})")
            
            # Initialize exchange
            exchange_class = getattr(ccxt, exchange)
            exchange_instance = exchange_class({
                'enableRateLimit': True,
            })
            
            if not exchange_instance.has['fetchOHLCV']:
                error = f"Exchange {exchange} does not support OHLCV data"
                logger.error(error)
                if progress_key:
                    self._update_progress(progress_key, "ccxt", "failed", 0, error)
                return 0, error
            
            # Calculate timeframe
            since = exchange_instance.milliseconds() - (days * 24 * 60 * 60 * 1000)
            timeframe = '1d'  # Daily candles
            
            # Fetch OHLCV data - limit to max_rows_per_symbol to prevent excessive data
            all_ohlcv = []
            current_since = since
            max_rows = settings.max_rows_per_symbol  # Enforce 10,000 row limit
            
            while current_since < exchange_instance.milliseconds() and len(all_ohlcv) < max_rows:
                try:
                    # Calculate remaining rows needed
                    remaining = max_rows - len(all_ohlcv)
                    fetch_limit = min(1000, remaining)  # Don't fetch more than needed
                    
                    ohlcv = exchange_instance.fetch_ohlcv(symbol, timeframe, since=current_since, limit=fetch_limit)
                    if not ohlcv:
                        break
                    
                    # Only add up to the limit
                    if len(all_ohlcv) + len(ohlcv) > max_rows:
                        needed = max_rows - len(all_ohlcv)
                        all_ohlcv.extend(ohlcv[:needed])
                        break  # Reached limit
                    else:
                        all_ohlcv.extend(ohlcv)
                    
                    # Check if we've reached the current time (newest available data)
                    latest_timestamp = ohlcv[-1][0]
                    current_time = exchange_instance.milliseconds()
                    # If latest timestamp is within 1 day of current time, we have newest data
                    if current_time - latest_timestamp < (24 * 60 * 60 * 1000):
                        logger.info(f"Reached newest available data for {symbol} (timestamp: {latest_timestamp})")
                        break
                    
                    current_since = ohlcv[-1][0] + 1  # Next timestamp
                    
                    if progress_key:
                        self._update_progress(
                            progress_key,
                            "ccxt",
                            "scraping",
                            len(all_ohlcv),
                            f"Fetched {len(all_ohlcv)} candles (limit: {max_rows})"
                        )
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"CCXT fetch error: {str(e)}")
                    break
            
            if not all_ohlcv:
                error = f"No data found for {symbol} on {exchange}"
                logger.warning(error)
                if progress_key:
                    self._update_progress(progress_key, "ccxt", "failed", 0, error)
                return 0, error
            
            # Convert to DataFrame and sort by timestamp (newest last)
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.sort_values('timestamp')  # Ensure chronological order
            
            # Verify data is recent (most recent timestamp should be within last 7 days)
            if not df.empty:
                most_recent = df['timestamp'].max()
                now = pd.Timestamp.now(tz='UTC')
                days_old = (now - most_recent).days
                
                if days_old > 7:
                    error = f"CCXT data is stale: most recent data is {days_old} days old (threshold: 7 days)"
                    logger.warning(error)
                    if progress_key:
                        self._update_progress(progress_key, "ccxt", "failed", 0, error)
                    return 0, error
                
                logger.info(f"CCXT data is recent: most recent timestamp is {most_recent} ({days_old} days old)")
            
            # Limit rows to max_rows_per_symbol (already limited during fetch, but double-check)
            max_rows = settings.max_rows_per_symbol
            if len(df) > max_rows:
                # Keep most recent rows
                df = df.tail(max_rows).sort_values('timestamp')
                logger.info(f"Limited CCXT data to {max_rows} most recent rows")
            
            # Prepare data
            rows_to_insert = []
            async for session in db.get_session():
                # Check existing data
                existing_query = select(OHLCV).where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.source == "ccxt"
                    )
                )
                existing_result = await session.execute(existing_query)
                existing = existing_result.scalars().all()
                existing_timestamps = {row.timestamp for row in existing}
                
                # Prepare new rows
                for _, row in df.iterrows():
                    if row['timestamp'] not in existing_timestamps:
                        rows_to_insert.append({
                            'symbol': symbol,
                            'timestamp': row['timestamp'],
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                            'source': 'ccxt'
                        })
                
                # Insert in batches
                if rows_to_insert:
                    batch_size = 1000
                    total_inserted = 0
                    for i in range(0, len(rows_to_insert), batch_size):
                        batch = rows_to_insert[i:i + batch_size]
                        ohlcv_objects = [OHLCV(**row) for row in batch]
                        session.add_all(ohlcv_objects)
                        await session.commit()
                        total_inserted += len(batch)
                        
                        if progress_key:
                            self._update_progress(
                                progress_key,
                                "ccxt",
                                "scraping",
                                total_inserted,
                                f"Inserted {total_inserted} rows"
                            )
                    
                    # Enforce rolling window after all inserts
                    await integrity_service.enforce_rolling_window(symbol)
                    # Update metadata (will create new session)
                    await integrity_service.update_metadata(symbol)
                
                break  # Exit session context
            
            logger.info(f"CCXT: Scraped {len(rows_to_insert)} rows for {symbol}")
            if progress_key:
                self._update_progress(progress_key, "ccxt", "completed", len(rows_to_insert))
            
            return len(rows_to_insert), None
            
        except ImportError:
            error = "ccxt not installed. Install with: pip install ccxt"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "ccxt", "failed", 0, error)
            return 0, error
        except Exception as e:
            error = f"CCXT scraping failed: {str(e)}"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "ccxt", "failed", 0, error)
            return 0, error
    
    async def scrape_custom(
        self,
        symbol: str,
        url: str,
        schema: Dict,
        progress_key: Optional[str] = None
    ) -> Tuple[int, Optional[str]]:
        """
        Scrape data from a custom URL with schema validation.
        
        Args:
            symbol: Trading symbol
            url: Custom data source URL
            schema: Schema mapping (e.g., {"timestamp": "date", "open": "open_price", ...})
            progress_key: Optional key for progress tracking
            
        Returns:
            Tuple of (rows_scraped, error_message)
        """
        try:
            import requests
            
            if progress_key:
                self._update_progress(progress_key, "custom", "starting", 0)
            
            logger.info(f"Scraping {symbol} from custom source: {url}")
            
            # Fetch data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse JSON or CSV
            if url.endswith('.csv') or 'text/csv' in response.headers.get('content-type', ''):
                df = pd.read_csv(url)
            else:
                data = response.json()
                df = pd.DataFrame(data)
            
            if df.empty:
                error = f"No data found at custom URL: {url}"
                logger.warning(error)
                if progress_key:
                    self._update_progress(progress_key, "custom", "failed", 0, error)
                return 0, error
            
            # Validate and map schema
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            mapped_df = pd.DataFrame()
            
            for field in required_fields:
                if field not in schema:
                    error = f"Schema missing required field: {field}"
                    logger.error(error)
                    if progress_key:
                        self._update_progress(progress_key, "custom", "failed", 0, error)
                    return 0, error
                
                source_field = schema[field]
                if source_field not in df.columns:
                    error = f"Source data missing column: {source_field}"
                    logger.error(error)
                    if progress_key:
                        self._update_progress(progress_key, "custom", "failed", 0, error)
                    return 0, error
                
                mapped_df[field] = df[source_field]
            
            # Normalize timestamps to UTC and sort by timestamp (newest last)
            mapped_df['timestamp'] = pd.to_datetime(mapped_df['timestamp'], utc=True)
            mapped_df = mapped_df.sort_values('timestamp')  # Ensure chronological order
            
            # Verify data is recent (most recent timestamp should be within last 7 days)
            if not mapped_df.empty:
                most_recent = mapped_df['timestamp'].max()
                now = pd.Timestamp.now(tz='UTC')
                days_old = (now - most_recent).days
                
                if days_old > 7:
                    error = f"Custom source data is stale: most recent data is {days_old} days old (threshold: 7 days)"
                    logger.warning(error)
                    if progress_key:
                        self._update_progress(progress_key, "custom", "failed", 0, error)
                    return 0, error
                
                logger.info(f"Custom source data is recent: most recent timestamp is {most_recent} ({days_old} days old)")
            
            # Limit rows to max_rows_per_symbol before processing
            max_rows = settings.max_rows_per_symbol
            if len(mapped_df) > max_rows:
                # Keep most recent rows
                mapped_df = mapped_df.tail(max_rows).sort_values('timestamp')
                logger.info(f"Limited custom source data to {max_rows} most recent rows")
            
            # Prepare data
            rows_to_insert = []
            async for session in db.get_session():
                # Check existing data
                existing_query = select(OHLCV).where(
                    and_(
                        OHLCV.symbol == symbol,
                        OHLCV.source == "custom"
                    )
                )
                existing_result = await session.execute(existing_query)
                existing = existing_result.scalars().all()
                existing_timestamps = {row.timestamp for row in existing}
                
                # Prepare new rows
                for _, row in mapped_df.iterrows():
                    if row['timestamp'] not in existing_timestamps:
                        rows_to_insert.append({
                            'symbol': symbol,
                            'timestamp': row['timestamp'],
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                            'source': 'custom'
                        })
                
                # Insert in batches
                if rows_to_insert:
                    batch_size = 1000
                    total_inserted = 0
                    for i in range(0, len(rows_to_insert), batch_size):
                        batch = rows_to_insert[i:i + batch_size]
                        ohlcv_objects = [OHLCV(**row) for row in batch]
                        session.add_all(ohlcv_objects)
                        await session.commit()
                        total_inserted += len(batch)
                        
                        if progress_key:
                            self._update_progress(
                                progress_key,
                                "custom",
                                "scraping",
                                total_inserted,
                                f"Inserted {total_inserted} rows"
                            )
                    
                    # Enforce rolling window after all inserts
                    await integrity_service.enforce_rolling_window(symbol)
                    # Update metadata (will create new session)
                    await integrity_service.update_metadata(symbol)
                
                break  # Exit session context
            
            logger.info(f"Custom source: Scraped {len(rows_to_insert)} rows for {symbol}")
            if progress_key:
                self._update_progress(progress_key, "custom", "completed", len(rows_to_insert))
            
            return len(rows_to_insert), None
            
        except ImportError:
            error = "requests not installed. Install with: pip install requests"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "custom", "failed", 0, error)
            return 0, error
        except Exception as e:
            error = f"Custom source scraping failed: {str(e)}"
            logger.error(error)
            if progress_key:
                self._update_progress(progress_key, "custom", "failed", 0, error)
            return 0, error
    
    async def scrape_multiple_sources(
        self,
        symbol: str,
        sources: List[str],
        source_params: Optional[Dict[str, Dict]] = None,
        progress_key: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Scrape from multiple sources simultaneously until minimum rows reached.
        
        Args:
            symbol: Trading symbol
            sources: List of sources to use (yahoo, alphavantage, ccxt, custom)
            source_params: Optional parameters for each source
            
        Returns:
            Dictionary with results from each source
        """
        if not progress_key:
            progress_key = f"{symbol}_{datetime.utcnow().isoformat()}"

        if progress_key not in self._progress:
            self._progress[progress_key] = {
                'symbol': symbol,
                'sources': {},
                'total_rows': 0,
                'status': 'running'
            }
        else:
            self._progress[progress_key]['symbol'] = symbol
            self._progress[progress_key]['status'] = 'running'
        
        if source_params is None:
            source_params = {}
        
        # Run all sources concurrently
        tasks = []
        task_sources = []
        
        if 'yahoo' in sources:
            params = source_params.get('yahoo', {})
            tasks.append(self.scrape_yahoo_finance(symbol, **params, progress_key=progress_key))
            task_sources.append('yahoo')
        
        if 'alphavantage' in sources:
            params = source_params.get('alphavantage', {})
            tasks.append(self.scrape_alpha_vantage(symbol, **params, progress_key=progress_key))
            task_sources.append('alphavantage')
        
        if 'ccxt' in sources:
            params = source_params.get('ccxt', {})
            # Convert symbol format for CCXT if needed (BTCUSDT -> BTC/USDT)
            ccxt_symbol = params.get('symbol', symbol)
            if '/' not in ccxt_symbol and len(ccxt_symbol) >= 6:
                # Try to split common crypto pairs
                if ccxt_symbol.endswith('USDT'):
                    base = ccxt_symbol[:-4]
                    ccxt_symbol = f"{base}/USDT"
                elif ccxt_symbol.endswith('USD'):
                    base = ccxt_symbol[:-3]
                    ccxt_symbol = f"{base}/USD"
            params['symbol'] = ccxt_symbol
            tasks.append(self.scrape_ccxt(ccxt_symbol, **params, progress_key=progress_key))
            task_sources.append('ccxt')
        
        if 'custom' in sources:
            params = source_params.get('custom', {})
            if 'url' in params and 'schema' in params:
                tasks.append(self.scrape_custom(symbol, **params, progress_key=progress_key))
                task_sources.append('custom')
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            source_name = task_sources[i] if i < len(task_sources) else 'unknown'
            if isinstance(result, Exception):
                error_msg = f"Source {source_name} task failed: {str(result)}"
                logger.error(error_msg)
                self._update_progress(progress_key, source_name, "failed", 0, error_msg)
            else:
                rows, error = result
                if error:
                    logger.warning(f"Source {source_name} failed: {error}")
                    self._update_progress(progress_key, source_name, "failed", int(rows or 0), error)
                else:
                    logger.info(f"Source {source_name} completed: {rows} rows")
                    self._update_progress(progress_key, source_name, "completed", int(rows or 0), f"Completed: {rows} rows")
        
        # Check total rows in database
        async for session in db.get_session():
            count_query = select(func.count(OHLCV.id)).where(OHLCV.symbol == symbol)
            count_result = await session.execute(count_query)
            total_rows = count_result.scalar() or 0
            break
        
        self._progress[progress_key]['total_rows'] = total_rows
        self._progress[progress_key]['status'] = 'completed' if total_rows >= settings.min_rows_per_symbol else 'insufficient'
        
        return self._progress[progress_key]
    
    async def append_live_mt5_data(self, symbol: str) -> Tuple[int, Optional[str]]:
        """
        Append live MT5/mock data to scraped dataset.
        
        Fetches recent OHLCV data from MT5 (or mock) and appends it to the database.
        This ensures scraped datasets are supplemented with the most recent live data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (rows_appended, error_message)
        """
        try:
            from app.services import get_mt5_client
            from app.services.integrity import integrity_service
            
            mt5_client = get_mt5_client()
            if not mt5_client.is_connected:
                return 0, "MT5 client not connected"
            
            logger.info(f"Appending live MT5 data for {symbol}")
            
            # Get recent OHLCV data (last 7 days, daily bars)
            from datetime import timedelta
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            ohlcv_df = mt5_client.get_ohlcv(symbol, timeframe='D1', start=start_time, end=end_time)
            
            if ohlcv_df.empty:
                logger.debug(f"No live OHLCV data available for {symbol}")
                return 0, None
            
            # Convert to list of dicts for append_live_data
            live_data = []
            for _, row in ohlcv_df.iterrows():
                live_data.append({
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'source': 'live_mt5'
                })
            
            # Append using integrity service
            rows_appended, error = await integrity_service.append_live_data(symbol, live_data)
            
            if error:
                logger.warning(f"Failed to append live data: {error}")
            else:
                logger.info(f"Appended {rows_appended} live MT5 rows for {symbol}")
            
            return rows_appended, error
            
        except Exception as e:
            error = f"Failed to append live MT5 data: {str(e)}"
            logger.error(error)
            return 0, error
    
    def get_progress(self, progress_key: str) -> Optional[Dict]:
        """Get scraping progress for a given key."""
        return self._progress.get(progress_key)
    
    def _update_progress(self, progress_key: str, source: str, status: str, rows: int, message: Optional[str] = None):
        """Update progress tracking."""
        if progress_key not in self._progress:
            self._progress[progress_key] = {'sources': {}}
        
        if 'sources' not in self._progress[progress_key]:
            self._progress[progress_key]['sources'] = {}
        
        self._progress[progress_key]['sources'][source] = {
            'status': status,
            'rows': rows,
            'message': message,
            'updated_at': datetime.utcnow().isoformat()
        }


# Global scraper instance
scraper_service = ScraperService()
