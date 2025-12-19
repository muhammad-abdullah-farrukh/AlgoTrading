"""Dataset integrity service for managing data quality and limits."""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, and_
from app.database import db
from app.models import OHLCV, DatasetMetadata
from app.config import settings
from app.utils.logging import get_logger
import json

logger = get_logger(__name__)


class IntegrityService:
    """Service for maintaining dataset integrity and rolling windows."""
    
    async def enforce_rolling_window(self, symbol: str) -> Tuple[int, Optional[str]]:
        """
        Enforce rolling window limit for a symbol by deleting oldest rows.
        
        Args:
            symbol: Trading symbol to enforce limit for
            
        Returns:
            Tuple of (rows_deleted, error_message)
        """
        try:
            async for session in db.get_session():
                # Count current rows
                count_query = select(func.count(OHLCV.id)).where(OHLCV.symbol == symbol)
                count_result = await session.execute(count_query)
                current_count = count_result.scalar() or 0
                
                if current_count <= settings.max_rows_per_symbol:
                    logger.debug(f"Symbol {symbol} within limit: {current_count}/{settings.max_rows_per_symbol}")
                    return 0, None
                
                # Calculate how many rows to delete
                rows_to_delete = current_count - settings.max_rows_per_symbol
                
                logger.info(
                    f"Enforcing rolling window for {symbol}: "
                    f"deleting {rows_to_delete} oldest rows "
                    f"(current: {current_count}, limit: {settings.max_rows_per_symbol})"
                )
                
                # Get oldest rows to delete (ordered by timestamp, then by id for consistency)
                delete_query = (
                    select(OHLCV.id)
                    .where(OHLCV.symbol == symbol)
                    .order_by(OHLCV.timestamp.asc(), OHLCV.id.asc())
                    .limit(rows_to_delete)
                )
                
                result = await session.execute(delete_query)
                ids_to_delete = [row[0] for row in result.all()]
                
                if ids_to_delete:
                    # Delete oldest rows
                    delete_stmt = delete(OHLCV).where(OHLCV.id.in_(ids_to_delete))
                    await session.execute(delete_stmt)
                    await session.commit()
                    
                    logger.info(f"Deleted {len(ids_to_delete)} oldest rows for {symbol}")
                    
                    # Update metadata (will create its own session)
                    await self.update_metadata(symbol)
                    
                    return len(ids_to_delete), None
                else:
                    return 0, None
                
                break  # Exit session context
                
        except Exception as e:
            error = f"Failed to enforce rolling window for {symbol}: {str(e)}"
            logger.error(error)
            return 0, error
    
    async def check_integrity(self, symbol: str) -> Dict:
        """
        Check dataset integrity for a symbol.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            Dictionary with integrity check results
        """
        try:
            async for session in db.get_session():
                # Count total rows
                total_query = select(func.count(OHLCV.id)).where(OHLCV.symbol == symbol)
                total_result = await session.execute(total_query)
                total_rows = total_result.scalar() or 0
                
                # Count rows per source
                source_query = (
                    select(OHLCV.source, func.count(OHLCV.id).label('count'))
                    .where(OHLCV.symbol == symbol)
                    .group_by(OHLCV.source)
                )
                source_result = await session.execute(source_query)
                rows_per_source = {row.source or 'unknown': row.count for row in source_result}
                
                # Get date range
                date_query = (
                    select(
                        func.min(OHLCV.timestamp).label('min_date'),
                        func.max(OHLCV.timestamp).label('max_date')
                    )
                    .where(OHLCV.symbol == symbol)
                )
                date_result = await session.execute(date_query)
                date_row = date_result.first()
                
                min_date = date_row.min_date if date_row else None
                max_date = date_row.max_date if date_row else None
                
                # Check if data is stale (most recent timestamp should be within last 7 days)
                is_stale = False
                days_since_last = None
                if max_date:
                    now = datetime.utcnow()
                    if max_date.tzinfo is None:
                        # Assume UTC if no timezone
                        max_date_utc = max_date.replace(tzinfo=None)
                        now_utc = now.replace(tzinfo=None)
                        days_since_last = (now_utc - max_date_utc).days
                    else:
                        days_since_last = (now - max_date.replace(tzinfo=None)).days
                    
                    is_stale = days_since_last > 7
                
                # Check for duplicates (same timestamp)
                duplicate_query = (
                    select(OHLCV.timestamp, func.count(OHLCV.id).label('count'))
                    .where(OHLCV.symbol == symbol)
                    .group_by(OHLCV.timestamp)
                    .having(func.count(OHLCV.id) > 1)
                )
                duplicate_result = await session.execute(duplicate_query)
                duplicates = duplicate_result.all()
                
                # Check for gaps (optional - can be expensive for large datasets)
                has_gaps = None
                if total_rows > 0 and total_rows < 10000:  # Only check for smaller datasets
                    # This is a simplified check - in production you might want more sophisticated gap detection
                    has_gaps = False
                
                # Check limits
                within_limit = total_rows <= settings.max_rows_per_symbol
                meets_minimum = total_rows >= settings.min_rows_per_symbol
                
                # Check data quality (null values)
                null_check_query = (
                    select(func.count(OHLCV.id))
                    .where(
                        and_(
                            OHLCV.symbol == symbol,
                            (
                                (OHLCV.open.is_(None)) |
                                (OHLCV.high.is_(None)) |
                                (OHLCV.low.is_(None)) |
                                (OHLCV.close.is_(None)) |
                                (OHLCV.volume.is_(None))
                            )
                        )
                    )
                )
                null_result = await session.execute(null_check_query)
                null_count = null_result.scalar() or 0
                
                integrity_status = {
                    'symbol': symbol,
                    'total_rows': total_rows,
                    'rows_per_source': rows_per_source,
                    'date_range': {
                        'start': min_date.isoformat() if min_date else None,
                        'end': max_date.isoformat() if max_date else None
                    },
                    'most_recent_age_days': days_since_last,
                    'is_stale': is_stale,
                    'duplicates': len(duplicates),
                    'duplicate_timestamps': [str(dup.timestamp) for dup in duplicates[:10]],  # First 10
                    'null_values': null_count,
                    'within_limit': within_limit,
                    'meets_minimum': meets_minimum,
                    'limit_max': settings.max_rows_per_symbol,
                    'limit_min': settings.min_rows_per_symbol,
                    'integrity_score': self._calculate_integrity_score(
                        total_rows, len(duplicates), null_count, within_limit, meets_minimum, is_stale
                    ),
                    'status': 'healthy' if within_limit and null_count == 0 and len(duplicates) == 0 and not is_stale else 'issues'
                }
                
                break  # Exit session context
                
            return integrity_status
            
        except Exception as e:
            logger.error(f"Integrity check failed for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'status': 'error'
            }
    
    async def update_metadata(self, symbol: str) -> Optional[str]:
        """
        Update dataset metadata for a symbol.
        
        Args:
            symbol: Trading symbol to update metadata for
            
        Returns:
            Error message if update failed, None otherwise
        """
        try:
            async for session in db.get_session():
                await self._update_metadata(symbol, session)
                break
            return None
        except Exception as e:
            error = f"Failed to update metadata for {symbol}: {str(e)}"
            logger.error(error)
            return error
    
    async def _update_metadata(self, symbol: str, session: AsyncSession) -> None:
        """Internal method to update metadata."""
        try:
            # Get current stats
            count_query = select(func.count(OHLCV.id)).where(OHLCV.symbol == symbol)
            count_result = await session.execute(count_query)
            row_count = count_result.scalar() or 0
            
            # Get date range
            date_query = (
                select(
                    func.min(OHLCV.timestamp).label('min_date'),
                    func.max(OHLCV.timestamp).label('max_date')
                )
                .where(OHLCV.symbol == symbol)
            )
            date_result = await session.execute(date_query)
            date_row = date_result.first()
            
            min_date = date_row.min_date if date_row else None
            max_date = date_row.max_date if date_row else None
            
            # Get sources
            source_query = (
                select(OHLCV.source, func.count(OHLCV.id).label('count'))
                .where(OHLCV.symbol == symbol)
                .group_by(OHLCV.source)
            )
            source_result = await session.execute(source_query)
            sources = {row.source or 'unknown': row.count for row in source_result}
            
            # Find or create metadata record
            metadata_name = f"dataset_{symbol}"
            metadata_query = select(DatasetMetadata).where(DatasetMetadata.name == metadata_name)
            metadata_result = await session.execute(metadata_query)
            metadata = metadata_result.scalar_one_or_none()
            
            if metadata:
                # Update existing
                metadata.row_count = row_count
                metadata.start_date = min_date
                metadata.end_date = max_date
                metadata.updated_at = datetime.utcnow()
                metadata.metadata_json = json.dumps({
                    'sources': sources,
                    'max_rows_limit': settings.max_rows_per_symbol,
                    'min_rows_target': settings.min_rows_per_symbol,
                    'last_updated': datetime.utcnow().isoformat()
                })
            else:
                # Create new
                metadata = DatasetMetadata(
                    name=metadata_name,
                    description=f"OHLCV dataset for {symbol}",
                    symbol=symbol,
                    start_date=min_date,
                    end_date=max_date,
                    row_count=row_count,
                    metadata_json=json.dumps({
                        'sources': sources,
                        'max_rows_limit': settings.max_rows_per_symbol,
                        'min_rows_target': settings.min_rows_per_symbol,
                        'created_at': datetime.utcnow().isoformat()
                    })
                )
                session.add(metadata)
            
            await session.commit()
            logger.debug(f"Updated metadata for {symbol}: {row_count} rows")
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            raise
    
    def _calculate_integrity_score(self, total_rows: int, duplicates: int, null_count: int, 
                                   within_limit: bool, meets_minimum: bool, is_stale: bool = False) -> float:
        """
        Calculate integrity score (0-100).
        
        Args:
            total_rows: Total number of rows
            duplicates: Number of duplicate timestamps
            null_count: Number of rows with null values
            within_limit: Whether within max limit
            meets_minimum: Whether meets minimum requirement
            is_stale: Whether data is stale (most recent > 7 days old)
            
        Returns:
            Integrity score from 0 to 100
        """
        score = 100.0
        
        # Deduct for stale data (most severe)
        if is_stale:
            score -= 40  # Significant deduction for stale data
        
        # Deduct for duplicates
        if total_rows > 0:
            duplicate_ratio = duplicates / total_rows
            score -= min(duplicate_ratio * 100, 30)  # Max 30 points deduction
        
        # Deduct for null values
        if total_rows > 0:
            null_ratio = null_count / total_rows
            score -= min(null_ratio * 100, 30)  # Max 30 points deduction
        
        # Deduct for limit violations
        if not within_limit:
            score -= 20
        
        # Deduct for not meeting minimum (less severe)
        if not meets_minimum and total_rows > 0:
            score -= 10
        
        return max(0.0, min(100.0, score))
    
    async def append_live_data(self, symbol: str, data: List[Dict]) -> Tuple[int, Optional[str]]:
        """
        Append live data to dataset (append-only operation).
        
        Args:
            symbol: Trading symbol
            data: List of OHLCV data dictionaries
            
        Returns:
            Tuple of (rows_inserted, error_message)
        """
        try:
            if not data:
                return 0, None
            
            async for session in db.get_session():
                # Check existing timestamps to prevent duplicates
                timestamps = [row['timestamp'] for row in data if 'timestamp' in row]
                if timestamps:
                    existing_query = select(OHLCV.timestamp).where(
                        and_(
                            OHLCV.symbol == symbol,
                            OHLCV.timestamp.in_(timestamps)
                        )
                    )
                    existing_result = await session.execute(existing_query)
                    existing_timestamps = {row[0] for row in existing_result.all()}
                else:
                    existing_timestamps = set()
                
                # Filter out duplicates
                new_data = [
                    row for row in data
                    if row.get('timestamp') not in existing_timestamps
                ]
                
                if not new_data:
                    logger.debug(f"No new data to append for {symbol}")
                    return 0, None
                
                # Insert new data
                ohlcv_objects = []
                for row in new_data:
                    ohlcv_objects.append(OHLCV(
                        symbol=symbol,
                        timestamp=row['timestamp'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        source=row.get('source', 'live')
                    ))
                
                session.add_all(ohlcv_objects)
                await session.commit()
                
                logger.info(f"Appended {len(ohlcv_objects)} new rows for {symbol}")
                
                # Enforce rolling window after append
                deleted, error = await self.enforce_rolling_window(symbol)
                if error:
                    logger.warning(f"Rolling window enforcement warning: {error}")
                
                # Update metadata
                await self._update_metadata(symbol, session)
                
                return len(ohlcv_objects), None
                
                break  # Exit session context
                
        except Exception as e:
            error = f"Failed to append live data for {symbol}: {str(e)}"
            logger.error(error)
            return 0, error
    
    async def get_dataset_stats(self, symbol: Optional[str] = None) -> Dict:
        """
        Get dataset statistics.
        
        Args:
            symbol: Optional symbol to filter by (None = all symbols)
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            async for session in db.get_session():
                if symbol:
                    # Single symbol stats
                    count_query = select(func.count(OHLCV.id)).where(OHLCV.symbol == symbol)
                    count_result = await session.execute(count_query)
                    total_rows = count_result.scalar() or 0
                    
                    symbols = [symbol]
                else:
                    # All symbols stats
                    symbol_query = (
                        select(OHLCV.symbol, func.count(OHLCV.id).label('count'))
                        .group_by(OHLCV.symbol)
                    )
                    symbol_result = await session.execute(symbol_query)
                    symbol_counts = {row.symbol: row.count for row in symbol_result}
                    
                    total_rows = sum(symbol_counts.values())
                    symbols = list(symbol_counts.keys())
                
                # Get metadata
                metadata_list = []
                for sym in symbols:
                    metadata_name = f"dataset_{sym}"
                    metadata_query = select(DatasetMetadata).where(DatasetMetadata.name == metadata_name)
                    metadata_result = await session.execute(metadata_query)
                    metadata = metadata_result.scalar_one_or_none()
                    
                    if metadata:
                        metadata_list.append({
                            'symbol': sym,
                            'row_count': metadata.row_count,
                            'start_date': metadata.start_date.isoformat() if metadata.start_date else None,
                            'end_date': metadata.end_date.isoformat() if metadata.end_date else None,
                            'updated_at': metadata.updated_at.isoformat() if metadata.updated_at else None
                        })
                
                return {
                    'total_rows': total_rows,
                    'symbols': symbols if not symbol else [symbol],
                    'metadata': metadata_list,
                    'limits': {
                        'max_per_symbol': settings.max_rows_per_symbol,
                        'min_per_symbol': settings.min_rows_per_symbol
                    }
                }
                
                break  # Exit session context
                
        except Exception as e:
            logger.error(f"Failed to get dataset stats: {str(e)}")
            return {'error': str(e)}


# Global integrity service instance
integrity_service = IntegrityService()
