import asyncio
from aiologger import Logger

async def test_logger():
    try:
        # Initialize logger with default handlers
        logger = Logger.with_default_handlers(name="test_logger")
        print(f"Logger initialized with default handlers: {logger}")

        # Log messages
        await logger.info("This is a test info log.")
        await logger.error("This is a test error log.")

        # Shutdown logger
        await logger.shutdown()
    except Exception as e:
        print(f"Error in test_logger: {e}")

asyncio.run(test_logger())