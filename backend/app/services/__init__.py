"""MT5 service with automatic fallback to mock."""
from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Check if MT5 is enabled in config
_mt5_enabled = settings.mt5_enabled

# Try to import real MT5 client
_mt5_module_available = False
RealMT5Client = None

if _mt5_enabled:
    try:
        import MetaTrader5 as mt5
        from app.services.mt5_client import MT5Client as RealMT5Client
        _mt5_module_available = True
        logger.info("MT5 module available")
    except (ImportError, Exception) as e:
        logger.info(f"MT5 module not available, will use mock client: {str(e)}")
        _mt5_module_available = False

# Import mock client (always available)
from app.services.mock_mt5 import MockMT5Client

# Singleton MT5 client instance
_mt5_client_instance = None
_mt5_client_initialized = False

# Export the available client
__all__ = ['get_mt5_client', 'initialize_mt5_client']


def initialize_mt5_client():
    """
    Initialize MT5 client singleton at application startup.
    
    This should be called once during application startup to create
    and connect the MT5 client instance that will be reused throughout
    the application lifecycle.
    
    Returns:
        MT5Client or MockMT5Client: Initialized MT5 client instance
    """
    global _mt5_client_instance, _mt5_client_initialized
    
    if _mt5_client_initialized:
        return _mt5_client_instance
    
    logger.info("Initializing MT5 client singleton...")
    
    if _mt5_enabled and _mt5_module_available and RealMT5Client is not None:
        try:
            client = RealMT5Client()
            # Try to connect (will fallback if connection fails)
            if client.connect(
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server
            ):
                logger.info("✓ Using real MT5 client (connected successfully)")
                _mt5_client_instance = client
                _mt5_client_initialized = True
                return client
            else:
                logger.info("MT5 connection failed, falling back to mock client")
                try:
                    client.shutdown()
                except Exception:
                    pass
        except ImportError:
            # MT5 module not installed - expected in some environments
            logger.info("MT5 module not installed, using mock client")
        except Exception as e:
            logger.info(f"MT5 client initialization failed, using mock: {str(e)}")
    
    # Use mock client
    logger.info("✓ Using mock MT5 client")
    mock_client = MockMT5Client()
    mock_client.connect()  # Mock connect always succeeds
    _mt5_client_instance = mock_client
    _mt5_client_initialized = True
    return mock_client


def get_mt5_client():
    """
    Get the singleton MT5 client instance.
    
    If not yet initialized, initializes it automatically (lazy initialization).
    For production use, call initialize_mt5_client() at startup instead.
    
    Returns:
        MT5Client or MockMT5Client: Singleton MT5 client instance
    """
    global _mt5_client_instance, _mt5_client_initialized
    
    if not _mt5_client_initialized:
        # Lazy initialization if not initialized at startup
        logger.debug("MT5 client not initialized at startup, initializing now...")
        return initialize_mt5_client()
    
    return _mt5_client_instance
