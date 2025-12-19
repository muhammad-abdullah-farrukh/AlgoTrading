"""Single entry point for running the application."""
import sys
import uvicorn
from app.config import settings
from app.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Run the FastAPI application."""
    try:
        logger.info("=" * 60)
        logger.info("Trading Application Backend - Starting Server")
        logger.info("=" * 60)
        logger.info(f"Host: {settings.host}")
        logger.info(f"Port: {settings.port}")
        logger.info(f"Reload: {settings.reload}")
        logger.info(f"Log Level: {settings.log_level}")
        logger.info("=" * 60)
        
        # Configure uvicorn for graceful shutdown
        config = uvicorn.Config(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level=settings.log_level.lower(),
            access_log=True,
            timeout_graceful_shutdown=5.0,  # Graceful shutdown timeout (seconds)
        )
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error starting server: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
