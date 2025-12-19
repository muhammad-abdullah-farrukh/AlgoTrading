"""FastAPI application main module."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import db
from app.routers import health
from app.routers.websocket import router as websocket_router
from app.routers import scraper, integrity, indicators, trading, autotrading, ml_training
from app.utils.logging import setup_logging, get_logger
import asyncio

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Initializes all components:
    - Database connection and tables
    - MT5 client (or mock fallback)
    - WebSocket manager
    - Scraper service
    - Trading services
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Trading Application Backend")
    logger.info("=" * 60)
    
    startup_errors = []
    
    # 0. Check scikit-learn availability (required for ML features)
    try:
        import sklearn
        logger.info(f"✓ scikit-learn available (version: {sklearn.__version__})")
    except ImportError:
        logger.warning("=" * 60)
        logger.warning("⚠ scikit-learn is NOT installed")
        logger.warning("=" * 60)
        logger.warning("ML features (model training, signal generation) will not be available.")
        logger.warning("To install scikit-learn, run:")
        logger.warning("  pip install scikit-learn")
        logger.warning("Or install all dependencies:")
        logger.warning("  pip install -r requirements.txt")
        logger.warning("=" * 60)
        startup_errors.append("scikit-learn not installed - ML features unavailable")
    
    # 1. Verify AI Dataset Pipeline (Phase ML-1)
    try:
        logger.info("Verifying AI dataset loading pipeline...")
        from app.ai import dataset_manager
        verification_results = dataset_manager.verify_dataset_pipeline()
        if verification_results['errors']:
            warning_msg = f"Dataset pipeline verification found {len(verification_results['errors'])} issue(s)"
            logger.warning(f"⚠ {warning_msg}")
            startup_errors.append(warning_msg)
        else:
            logger.info("✓ AI dataset pipeline verified successfully")
    except ImportError as e:
        # If sklearn import fails in dataset_manager, it's because of ML model imports
        if 'sklearn' in str(e) or 'scikit-learn' in str(e):
            logger.warning("⚠ AI dataset pipeline verification skipped (scikit-learn not available)")
            logger.warning("  Dataset loading works, but ML model features require scikit-learn")
        else:
            error_msg = f"AI dataset pipeline verification failed: {str(e)}"
            logger.warning(f"⚠ {error_msg}")
            startup_errors.append(error_msg)
    except Exception as e:
        error_msg = f"AI dataset pipeline verification failed: {str(e)}"
        logger.warning(f"⚠ {error_msg}")
        startup_errors.append(error_msg)
        # Don't fail startup - dataset pipeline is optional for basic operation
    
    # 1b. Verify AI Model Integration (Phase V-1)
    try:
        logger.info("Verifying AI model integration...")
        from app.ai.models.logistic_regression import logistic_model
        from app.ai.signal_generator import signal_generator
        from app.ai.model_export import model_export_service
        
        # Check if models directory exists and has model files
        from pathlib import Path
        current_file = Path(__file__).resolve()
        models_dir = current_file.parent / "ai" / "models"
        model_files = list(models_dir.glob("logistic_regression_*.pkl"))
        
        if model_files:
            logger.info(f"✓ Found {len(model_files)} trained model(s)")
            # Try to load the latest model to verify it works
            latest_model = sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            if logistic_model.load_model(latest_model):
                metadata = logistic_model.get_metadata()
                if metadata:
                    accuracy = metadata.get('accuracy', 'N/A')
                    timeframe = metadata.get('timeframe', 'unknown')
                    logger.info(f"✓ Model loaded successfully")
                    logger.info(f"  Timeframe: {timeframe}")
                    logger.info(f"  Accuracy: {accuracy}")
                    logger.info(f"  Sample Size: {metadata.get('sample_size', 'N/A')}")
                    logger.info(f"  Last Trained: {metadata.get('timestamp', 'N/A')}")
                else:
                    logger.warning("⚠ Model loaded but metadata not found")
            else:
                logger.warning("⚠ Failed to load latest model (may be corrupted)")
        else:
            logger.info("ℹ No trained models found (this is normal if training hasn't been performed)")
            logger.info("  To train a model, run: python train_model.py 1d")
        
        logger.info("✓ AI model integration verified")
    except ImportError as e:
        if 'sklearn' in str(e) or 'scikit-learn' in str(e):
            logger.warning("⚠ AI model integration verification skipped (scikit-learn not available)")
        else:
            error_msg = f"AI model integration verification failed: {str(e)}"
            logger.warning(f"⚠ {error_msg}")
            startup_errors.append(error_msg)
    except Exception as e:
        error_msg = f"AI model integration verification failed: {str(e)}"
        logger.warning(f"⚠ {error_msg}")
        startup_errors.append(error_msg)
    
    # 2. Initialize Database
    try:
        logger.info("Initializing database...")
        await db.connect()
        logger.info("✓ Database connected and tables initialized")
    except Exception as e:
        error_msg = f"Database initialization failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        startup_errors.append(error_msg)
        # Don't raise - allow other components to initialize
    
    # 3. Initialize MT5 Client (or mock) - Singleton instance
    try:
        logger.info("Initializing MT5 client...")
        from app.services import initialize_mt5_client
        from app.services.mt5_status import mt5_status_service
        from app.config import settings
        
        mt5_client = initialize_mt5_client()
        
        # Determine connection mode and persist status
        try:
            from app.services.mock_mt5 import MockMT5Client
            is_mock = isinstance(mt5_client, MockMT5Client)
            
            if mt5_client.is_connected:
                if is_mock:
                    logger.info("✓ Using mock MT5 client")
                    await mt5_status_service.update_status(
                        connected=True,
                        mode="mock"
                    )
                else:
                    logger.info("✓ MT5 client connected (real)")
                    # Get login/server from client if available
                    login = getattr(mt5_client, '_login', None)
                    server = getattr(mt5_client, '_server', None)
                    await mt5_status_service.update_status(
                        connected=True,
                        mode="real",
                        login=login,
                        server=server
                    )
                logger.info("✓ MT5 connection status persisted to database")
            else:
                logger.info("✓ MT5 client not connected")
                await mt5_status_service.update_status(
                    connected=False,
                    mode="mock"
                )
                logger.info("✓ MT5 connection status persisted to database")
        except Exception as e:
            logger.warning(f"⚠ Failed to persist MT5 connection status: {str(e)}")
    except Exception as e:
        error_msg = f"MT5 client initialization warning: {str(e)}"
        logger.warning(f"⚠ {error_msg}")
        # Non-critical, continue
    
    # 3. Initialize WebSocket Manager
    try:
        logger.info("Initializing WebSocket manager...")
        from app.websocket import manager
        logger.info(f"✓ WebSocket manager ready (connections: {manager.get_connection_count()})")
    except Exception as e:
        error_msg = f"WebSocket manager initialization failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        startup_errors.append(error_msg)
    
    # 5. Initialize Scraper Service
    try:
        logger.info("Initializing scraper service...")
        from app.services.scraper import scraper_service
        logger.info("✓ Scraper service ready")
    except Exception as e:
        error_msg = f"Scraper service initialization failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        startup_errors.append(error_msg)
    
    # 5. Initialize Trading Services
    try:
        logger.info("Initializing trading services...")
        from app.services.paper_trading import paper_trading_service
        from app.services.autotrading import autotrading_service
        logger.info("✓ Paper trading service ready")
        logger.info("✓ Autotrading service ready")
    except Exception as e:
        error_msg = f"Trading services initialization failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        startup_errors.append(error_msg)
    
    # 6. Initialize Indicators Service
    try:
        logger.info("Initializing indicators service...")
        from app.services.indicators import indicators_service
        logger.info("✓ Indicators service ready")
    except Exception as e:
        error_msg = f"Indicators service initialization failed: {str(e)}"
        logger.error(f"✗ {error_msg}")
        startup_errors.append(error_msg)
    
    # Summary
    logger.info("=" * 60)
    if startup_errors:
        logger.warning(f"Application started with {len(startup_errors)} warning(s)")
        for error in startup_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("✓ All components initialized successfully")
    logger.info(f"Application ready on {settings.host}:{settings.port}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down application...")
    logger.info("=" * 60)
    
    shutdown_errors = []
    shutdown_timeout = 5.0  # Maximum time for shutdown (seconds)
    
    # 1. Disconnect WebSocket connections (with timeout)
    # Do this first to allow WebSocket loops to exit cleanly
    try:
        logger.info("Closing WebSocket connections...")
        from app.websocket import manager
        connection_count = manager.get_connection_count()
        if connection_count > 0:
            logger.info(f"  Closing {connection_count} active WebSocket connections...")
            await asyncio.wait_for(
                manager.shutdown_all(timeout=shutdown_timeout * 0.4),  # Use 40% of timeout
                timeout=shutdown_timeout * 0.4
            )
        logger.info("[OK] WebSocket connections closed")
    except asyncio.CancelledError:
        # Normal cancellation during shutdown - don't log as error
        logger.debug("WebSocket shutdown cancelled (normal during app shutdown)")
    except asyncio.TimeoutError:
        error_msg = "WebSocket shutdown timed out"
        logger.warning(f"[WARNING] {error_msg}")
        shutdown_errors.append(error_msg)
    except Exception as e:
        error_msg = f"WebSocket shutdown error: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        shutdown_errors.append(error_msg)
    
    # 2. Cancel all remaining pending asyncio tasks (except current one)
    # This ensures any remaining tasks (e.g., background tasks) are cancelled
    try:
        logger.info("Cancelling remaining pending tasks...")
        loop = asyncio.get_event_loop()
        pending_tasks = [t for t in asyncio.all_tasks(loop) if not t.done() and t != asyncio.current_task()]
        
        if pending_tasks:
            logger.info(f"  Cancelling {len(pending_tasks)} remaining tasks...")
            for task in pending_tasks:
                task.cancel()
            
            # Wait for tasks to finish cancellation (with timeout)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=shutdown_timeout * 0.2  # Use 20% of timeout for task cancellation
                )
            except asyncio.CancelledError:
                # Normal cancellation during shutdown
                logger.debug("Task cancellation cancelled (normal during shutdown)")
            except asyncio.TimeoutError:
                logger.warning("[WARNING] Some tasks did not cancel within timeout")
            except Exception as e:
                logger.debug(f"Error during task cancellation: {str(e)}")
        
        logger.info("[OK] Remaining tasks cancelled")
    except asyncio.CancelledError:
        # Normal cancellation during shutdown
        logger.debug("Task cancellation cancelled (normal during shutdown)")
    except Exception as e:
        error_msg = f"Task cancellation error: {str(e)}"
        logger.warning(f"[WARNING] {error_msg}")
        shutdown_errors.append(error_msg)
    
    # 3. Shutdown MT5 Client (with timeout)
    try:
        logger.info("Shutting down MT5 client...")
        from app.services import get_mt5_client
        mt5_client = get_mt5_client()
        if mt5_client and mt5_client.is_connected:
            # Run shutdown in executor to avoid blocking (MT5 shutdown may be synchronous)
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, mt5_client.shutdown),
                timeout=shutdown_timeout * 0.2  # Use 20% of timeout
            )
        logger.info("[OK] MT5 client shutdown")
    except asyncio.CancelledError:
        # Normal cancellation during shutdown
        logger.debug("MT5 shutdown cancelled (normal during app shutdown)")
    except asyncio.TimeoutError:
        error_msg = "MT5 client shutdown timed out"
        logger.warning(f"[WARNING] {error_msg}")
        shutdown_errors.append(error_msg)
    except Exception as e:
        error_msg = f"MT5 client shutdown error: {str(e)}"
        logger.warning(f"[WARNING] {error_msg}")
        shutdown_errors.append(error_msg)
    
    # 4. Disconnect Database (with timeout)
    try:
        logger.info("Disconnecting from database...")
        await asyncio.wait_for(
            db.disconnect(),
            timeout=shutdown_timeout * 0.2  # Use 20% of timeout
        )
        logger.info("[OK] Database disconnected")
    except asyncio.CancelledError:
        # Normal cancellation during shutdown
        logger.debug("Database disconnect cancelled (normal during app shutdown)")
    except asyncio.TimeoutError:
        error_msg = "Database disconnect timed out"
        logger.warning(f"[WARNING] {error_msg}")
        shutdown_errors.append(error_msg)
    except Exception as e:
        error_msg = f"Database shutdown error: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        shutdown_errors.append(error_msg)
    
    # Summary
    logger.info("=" * 60)
    if shutdown_errors:
        logger.warning(f"Application shutdown completed with {len(shutdown_errors)} warning(s)")
    else:
        logger.info("[OK] Application shutdown complete")
    logger.info("=" * 60)


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom exception handler for HTTP requests.
    
    Handles WebSocket endpoint HTTP requests gracefully and provides
    consistent error responses for all HTTPExceptions.
    """
    # Check if this is a request to a WebSocket path
    if request.url.path.startswith("/ws/"):
        # Log at INFO level instead of ERROR for expected behavior
        logger.info(
            f"HTTP request to WebSocket endpoint: {request.url.path} "
            f"(This is expected - WebSocket endpoints require ws:// protocol)"
        )
        # Return the HTTPException response as-is (already contains helpful error message)
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
        )
    
    # For other HTTPExceptions, use default behavior with logging
    logger.debug(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    
    Prevents crashes and provides meaningful error responses.
    """
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please check server logs.",
            "type": type(exc).__name__
        }
    )


# Include routers
# Health endpoint at root level
app.include_router(health.router)
# WebSocket endpoints
app.include_router(websocket_router)
# Scraping endpoints
app.include_router(scraper.router)
# Integrity endpoints
app.include_router(integrity.router)
# Indicators endpoints
app.include_router(indicators.router)
# Trading endpoints
app.include_router(trading.router)
# Autotrading endpoints
app.include_router(autotrading.router)
# ML Training endpoints
app.include_router(ml_training.router)
