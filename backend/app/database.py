"""Database configuration and connection management."""
from typing import AsyncGenerator

# Early import check for aiosqlite with clear error message
try:
    import aiosqlite
except ImportError:
    raise ImportError(
        "aiosqlite is required but not installed. "
        "Please install it with: pip install aiosqlite\n"
        "Or install all dependencies: pip install -r requirements.txt"
    )

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
# Note: Base is imported inside connect() method to avoid scoping issues with nested functions
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Database connection manager with SQLite."""
    
    def __init__(self):
        self._connected: bool = False
        self._engine = None
        self._session_factory = None
    
    async def connect(self) -> None:
        """Establish database connection and initialize tables."""
        if not self._connected:
            try:
                logger.info("Connecting to database...")
                logger.info(f"Database URL: {settings.database_url}")
                
                # Verify database URL format
                if not settings.database_url.startswith("sqlite+aiosqlite:///"):
                    error_msg = (
                        f"Invalid database URL format. Expected 'sqlite+aiosqlite:///' but got: "
                        f"{settings.database_url}\n"
                        f"Please set DATABASE_URL environment variable or update config.py"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Create async engine for SQLite
                try:
                    self._engine = create_async_engine(
                        settings.database_url,
                        echo=False,  # Set to True for SQL query logging
                        future=True
                    )
                    logger.info("Async engine created successfully")
                except Exception as e:
                    logger.error(f"Failed to create async engine: {str(e)}")
                    logger.error("Make sure 'aiosqlite' is installed: pip install aiosqlite")
                    raise
                
                # Create session factory
                try:
                    from sqlalchemy.ext.asyncio import async_sessionmaker  # type: ignore

                    self._session_factory = async_sessionmaker(
                        self._engine,
                        class_=AsyncSession,
                        expire_on_commit=False
                    )
                except Exception:
                    self._session_factory = sessionmaker(
                        bind=self._engine,
                        class_=AsyncSession,
                        expire_on_commit=False
                    )
                logger.info("Session factory created successfully")
                
                # Initialize database tables
                try:
                    # Import Base here to ensure it's available in the closure
                    # This prevents "Base not associated with a value" errors
                    from app.models import Base as BaseModel
                    
                    # Create a wrapper function that properly captures Base in closure
                    # This ensures Base is accessible when run_sync executes in sync context
                    def create_all_tables(connection):
                        """Create all tables using Base metadata."""
                        # Use BaseModel (captured from outer scope) to avoid scoping issues
                        BaseModel.metadata.create_all(connection)
                    
                    async with self._engine.begin() as conn:
                        await conn.run_sync(create_all_tables)
                    
                    # Run schema migrations for existing tables
                    await self._migrate_schema()
                    
                    # Log all created tables for verification
                    # Use BaseModel (imported above) to avoid scoping issues
                    table_names = list(BaseModel.metadata.tables.keys())
                    logger.info(f"Database tables initialized successfully: {', '.join(sorted(table_names))}")
                    logger.info(f"Total tables created: {len(table_names)}")
                    
                    # Verify critical tables exist
                    critical_tables = ['trades', 'positions', 'autotrading_settings', 'strategies', 'mt5_connection_status', 'trading_settings', 'ohlcv', 'dataset_metadata']
                    missing_tables = [t for t in critical_tables if t not in table_names]
                    if missing_tables:
                        logger.warning(f"Missing critical tables: {', '.join(missing_tables)}")
                    else:
                        logger.info("✓ All critical tables verified: trades, positions, autotrading_settings, strategies, mt5_connection_status, trading_settings, ohlcv, dataset_metadata")
                        
                except Exception as e:
                    logger.error(f"Failed to initialize database tables: {str(e)}")
                    raise
                
                logger.info("✓ Database initialized successfully")
                logger.info(f"✓ Database connected successfully: {settings.database_url}")
                self._connected = True
            except ImportError as e:
                logger.error(f"Missing required dependency: {str(e)}")
                logger.error("Please install dependencies: pip install -r requirements.txt")
                raise
            except Exception as e:
                logger.error(f"Database connection failed: {str(e)}")
                logger.error("Check database URL and ensure all dependencies are installed")
                raise
    
    async def _migrate_schema(self) -> None:
        """
        Migrate database schema to match current models.
        Adds missing columns to existing tables safely.
        """
        try:
            async with self._engine.begin() as conn:
                # Check and add missing columns to autotrading_settings
                migration_result = await conn.run_sync(self._migrate_autotrading_settings)
                migration_result_ohlcv = await conn.run_sync(self._migrate_ohlcv)

                migration_results = [r for r in [migration_result, migration_result_ohlcv] if r]
                if migration_results:
                    logger.info(f"✓ Schema migration completed: {'; '.join(migration_results)}")
                else:
                    logger.debug("Schema migration: no changes needed")
        except Exception as e:
            logger.warning(f"Schema migration warning: {str(e)}")
            # Don't fail startup if migration has issues - log and continue
    
    @staticmethod
    def _migrate_autotrading_settings(conn):
        """
        Migrate autotrading_settings table to add missing columns.
        This is a synchronous function that runs in the sync context.
        
        Returns:
            str: Description of migration actions taken, or None if no migration needed
        """
        from sqlalchemy import text, inspect
        
        try:
            # Check if table exists using inspector
            inspector = inspect(conn)
            table_names = inspector.get_table_names()
            
            if 'autotrading_settings' not in table_names:
                return None  # Table doesn't exist yet, create_all will handle it
            
            # Get existing columns
            existing_columns = {col['name'] for col in inspector.get_columns('autotrading_settings')}
            logger.debug(f"Existing columns in autotrading_settings: {sorted(existing_columns)}")
            
            # Define required columns to add (matching SQLAlchemy model)
            required_columns = {
                'selected_strategy_id': 'INTEGER',  # SQLite doesn't support FK in ALTER TABLE, but model has FK constraint
                'timeframe': 'VARCHAR(20)'
            }
            
            # Track what was added
            added_columns = []
            
            # Add missing columns
            for column_name, column_type in required_columns.items():
                if column_name not in existing_columns:
                    try:
                        # SQLite ALTER TABLE ADD COLUMN syntax
                        sql = f"ALTER TABLE autotrading_settings ADD COLUMN {column_name} {column_type}"
                        conn.execute(text(sql))
                        added_columns.append(column_name)
                        logger.info(f"✓ Added column '{column_name}' ({column_type}) to autotrading_settings table")
                    except Exception as e:
                        # Column might already exist or there's another issue
                        error_msg = str(e)
                        if 'duplicate column' in error_msg.lower() or 'already exists' in error_msg.lower():
                            logger.debug(f"Column '{column_name}' already exists, skipping")
                        else:
                            logger.error(f"Failed to add column '{column_name}': {error_msg}")
                            # Log error but don't fail - allow startup to continue
                            # The error will be visible in logs for manual intervention
                else:
                    logger.debug(f"Column '{column_name}' already exists in autotrading_settings")
            
            if added_columns:
                return f"Added columns: {', '.join(added_columns)}"
            return None
            
        except Exception as e:
            error_msg = f"Schema migration check failed: {str(e)}"
            logger.error(error_msg)
            # Log error but return None - don't fail startup
            # Migration errors should be investigated but shouldn't block application startup
            return None

    @staticmethod
    def _migrate_ohlcv(conn):
        """Migrate ohlcv table to add missing columns safely."""
        from sqlalchemy import text, inspect

        try:
            inspector = inspect(conn)
            table_names = inspector.get_table_names()
            if 'ohlcv' not in table_names:
                return None

            existing_columns = {col['name'] for col in inspector.get_columns('ohlcv')}
            logger.debug(f"Existing columns in ohlcv: {sorted(existing_columns)}")

            added_columns = []

            if 'source' not in existing_columns:
                try:
                    conn.execute(text("ALTER TABLE ohlcv ADD COLUMN source VARCHAR(100)"))
                    added_columns.append('source')
                    logger.info("✓ Added column 'source' (VARCHAR(100)) to ohlcv table")
                except Exception as e:
                    error_msg = str(e)
                    if 'duplicate column' in error_msg.lower() or 'already exists' in error_msg.lower():
                        logger.debug("Column 'source' already exists, skipping")
                    else:
                        logger.error(f"Failed to add column 'source' to ohlcv: {error_msg}")

            if added_columns:
                return f"ohlcv: added columns: {', '.join(added_columns)}"
            return None

        except Exception as e:
            logger.error(f"OHLCV schema migration check failed: {str(e)}")
            return None
    
    async def disconnect(self) -> None:
        """Close database connection and cleanup all sessions."""
        if self._connected:
            try:
                logger.info("Disconnecting from database...")
                if self._engine:
                    await self._engine.dispose()
                self._connected = False
                logger.info("Database disconnected successfully")
            except Exception as e:
                logger.error(f"Error during database disconnection: {str(e)}")
                self._connected = False
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session.
        
        Yields:
            AsyncSession: Database session
        """
        if not self._session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected


# Global database instance
db = Database()
