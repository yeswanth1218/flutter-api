import logging

def setup_logger(name=__name__, level=logging.INFO):
    """
    Set up a logger with console handler only.
    
    Args:
        name (str): Name of the logger (usually __name__)
        level (int): Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create console formatter
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add only console handler to logger
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name=__name__):
    """
    Get a configured logger instance.
    
    Args:
        name (str): Name of the logger (usually __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return setup_logger(name)

# Create a default logger for the application
app_logger = get_logger('business_card_api')

def log_request_info(request, endpoint_name):
    """
    Log incoming request information.
    
    Args:
        request: Flask request object
        endpoint_name (str): Name of the endpoint being called
    """
    logger = get_logger('request_logger')
    logger.info(f"Incoming {request.method} request to {endpoint_name}")
    logger.debug(f"Request headers: {dict(request.headers)}")
    logger.debug(f"Request args: {dict(request.args)}")
    if request.is_json:
        logger.debug(f"Request JSON: {request.get_json()}")

def log_response_info(response_data, status_code, endpoint_name):
    """
    Log outgoing response information.
    
    Args:
        response_data: Response data
        status_code (int): HTTP status code
        endpoint_name (str): Name of the endpoint
    """
    logger = get_logger('response_logger')
    logger.info(f"Response from {endpoint_name}: Status {status_code}")
    if status_code >= 400:
        logger.error(f"Error response from {endpoint_name}: {response_data}")
    else:
        logger.debug(f"Success response from {endpoint_name}")

def log_database_operation(operation, table=None, success=True, error=None):
    """
    Log database operations.
    
    Args:
        operation (str): Type of database operation (SELECT, INSERT, UPDATE, DELETE)
        table (str): Table name (optional)
        success (bool): Whether the operation was successful
        error (str): Error message if operation failed
    """
    logger = get_logger('database_logger')
    if success:
        logger.info(f"Database {operation} operation successful" + (f" on table {table}" if table else ""))
    else:
        logger.error(f"Database {operation} operation failed" + (f" on table {table}" if table else "") + f": {error}")

def log_llm_operation(operation, model=None, success=True, error=None, processing_time=None):
    """
    Log LLM operations.
    
    Args:
        operation (str): Type of LLM operation
        model (str): Model name used
        success (bool): Whether the operation was successful
        error (str): Error message if operation failed
        processing_time (float): Time taken for the operation in seconds
    """
    logger = get_logger('llm_logger')
    time_info = f" (took {processing_time:.2f}s)" if processing_time else ""
    model_info = f" using {model}" if model else ""
    
    if success:
        logger.info(f"LLM {operation} operation successful{model_info}{time_info}")
    else:
        logger.error(f"LLM {operation} operation failed{model_info}{time_info}: {error}")