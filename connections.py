import os
import psycopg2
from dotenv import load_dotenv
from logger_config import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)
logger.info("Database connections module initialized")

def get_db_connection():
    """Get PostgreSQL database connection."""
    logger.debug("Attempting to establish database connection")
    
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        logger.info("Database connection established successfully")
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None