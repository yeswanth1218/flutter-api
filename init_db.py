import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_database():
    """Initialize the database schema and tables."""
    print("=== Database Initialization Script ===")
    
    # Retrieve DB environment variables
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    database = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    schema = os.getenv('DB_SCHEMA', 'smart_stack')
    
    # Check if necessary variables are provided
    if not all([host, port, database, user, password]):
        print("Error: Missing database connection details in environment variables.")
        print("Please check that DB_HOST, DB_PORT, DB_NAME, DB_USER, and DB_PASSWORD are set in your .env file.")
        sys.exit(1)
        
    print(f"Connecting to database '{database}' on host '{host}:{port}' as user '{user}'...")
    
    try:
        # Establish connection (using default search path first to create the schema)
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        conn.autocommit = False
        cursor = conn.cursor()
        
        # 1. Create the schema if it doesn't exist
        print(f"Creating schema '{schema}' if it doesn't exist...")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
        
        # 2. Set search path to the schema
        print(f"Setting session search_path to '{schema}'...")
        cursor.execute(f"SET search_path TO {schema};")
        
        # 3. Create 'users' table
        print("Creating 'users' table if it doesn't exist...")
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(36) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            phone VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            status VARCHAR(50) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_users_table)
        
        # 4. Create 'cards' table
        print("Creating 'cards' table if it doesn't exist...")
        create_cards_table = """
        CREATE TABLE IF NOT EXISTS cards (
            card_id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(user_id) ON DELETE CASCADE,
            name VARCHAR(255),
            job_title VARCHAR(255),
            company VARCHAR(255),
            phone VARCHAR(255),
            email VARCHAR(255),
            website VARCHAR(255),
            address TEXT,
            linkedin VARCHAR(255),
            twitter VARCHAR(255),
            facebook VARCHAR(255),
            instagram VARCHAR(255),
            additional_info TEXT,
            tags TEXT[],
            card_type VARCHAR(50) DEFAULT 'business',
            status INTEGER DEFAULT 0,
            fav INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        );
        """
        cursor.execute(create_cards_table)
        
        # 5. Create 'categories' table
        print("Creating 'categories' table if it doesn't exist...")
        create_categories_table = """
        CREATE TABLE IF NOT EXISTS categories (
            user_id VARCHAR(36) REFERENCES users(user_id) ON DELETE CASCADE,
            category_name VARCHAR(255),
            status INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, category_name)
        );
        """
        cursor.execute(create_categories_table)
        
        # Commit transaction
        conn.commit()
        print(f"Database schema '{schema}' and all tables initialized successfully!")
        
    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()
        print(f"Error during database initialization: {e}")
        sys.exit(1)
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    init_database()
