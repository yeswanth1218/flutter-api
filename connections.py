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
    
    schema = os.getenv('DB_SCHEMA', 'smart_stack')
    
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        # Configure search path to the desired schema
        cursor = connection.cursor()
        cursor.execute(f"SET search_path TO {schema};")
        cursor.close()
        connection.commit()
        
        logger.info(f"Database connection established successfully with search_path='{schema}'")
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None