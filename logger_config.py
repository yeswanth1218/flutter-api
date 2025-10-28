import logging
import json

def format_json_data(data, max_length=500):
    """
    Format JSON data for structured logging with proper indentation and truncation.
    
    Args:
        data: Data to format (dict, list, or other)
        max_length (int): Maximum length for the formatted string
        
    Returns:
        str: Formatted JSON string
    """
    try:
        if isinstance(data, (dict, list)):
            # Pretty print JSON with indentation
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            
            # If the formatted JSON is too long, truncate it
            if len(formatted) > max_length:
                # Try to find a good truncation point (end of a complete object/array)
                truncated = formatted[:max_length]
                last_brace = max(truncated.rfind('}'), truncated.rfind(']'))
                if last_brace > max_length * 0.7:  # If we found a good truncation point
                    truncated = formatted[:last_brace + 1]
                else:
                    truncated = formatted[:max_length - 10]
                
                # Count remaining items
                remaining_chars = len(formatted) - len(truncated)
                truncated += f"\n... ({remaining_chars} more characters)"
                
            return formatted if len(formatted) <= max_length else truncated
        else:
            return str(data)
    except Exception as e:
        return f"<Error formatting data: {str(e)}>"

def log_separator(logger, title="", char="=", width=80):
    """
    Log a separator line for better visual organization.
    
    Args:
        logger: Logger instance
        title (str): Optional title to include in separator
        char (str): Character to use for separator
        width (int): Width of separator line
    """
    if title:
        title_formatted = f" {title} "
        padding = (width - len(title_formatted)) // 2
        separator = char * padding + title_formatted + char * padding
        if len(separator) < width:
            separator += char * (width - len(separator))
    else:
        separator = char * width
    
    logger.info(separator)

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
    Log incoming request information in a structured format.
    
    Args:
        request: Flask request object
        endpoint_name (str): Name of the endpoint being called
    """
    logger = get_logger('request_logger')
    
    # Log request separator
    log_separator(logger, f"INCOMING REQUEST: {endpoint_name.upper()}", "=", 60)
    
    logger.info(f"📥 Method: {request.method}")
    logger.info(f"📍 Endpoint: {endpoint_name}")
    logger.info(f"🌐 URL: {request.url}")
    
    if request.headers:
        logger.info("📋 Headers:")
        for key, value in dict(request.headers).items():
            # Hide sensitive headers
            if key.lower() in ['authorization', 'cookie', 'x-api-key']:
                value = "***HIDDEN***"
            logger.info(f"   {key}: {value}")
    
    if request.args:
        logger.info("🔍 Query Parameters:")
        for key, value in dict(request.args).items():
            logger.info(f"   {key}: {value}")
    
    if request.is_json:
        try:
            json_data = request.get_json(silent=True)
            if json_data:
                logger.info("📦 Request Body:")
                formatted_json = format_json_data(json_data, max_length=300)
                for line in formatted_json.split('\n'):
                    logger.info(f"   {line}")
        except Exception as e:
            logger.debug(f"Could not parse JSON from request: {e}")
    
    log_separator(logger, "", "-", 60)

def log_response_info(response_data, status_code, endpoint_name):
    """
    Log outgoing response information in a structured format.
    
    Args:
        response_data: Response data
        status_code (int): HTTP status code
        endpoint_name (str): Name of the endpoint
    """
    logger = get_logger('response_logger')
    
    # Determine response type and emoji
    if status_code >= 500:
        status_emoji = "💥"
        status_type = "SERVER ERROR"
        log_level = "error"
    elif status_code >= 400:
        status_emoji = "⚠️"
        status_type = "CLIENT ERROR"
        log_level = "error"
    elif status_code >= 300:
        status_emoji = "🔄"
        status_type = "REDIRECT"
        log_level = "info"
    else:
        status_emoji = "✅"
        status_type = "SUCCESS"
        log_level = "info"
    
    # Log response separator
    log_separator(logger, f"RESPONSE: {endpoint_name.upper()} - {status_type}", "=", 60)
    
    logger.info(f"{status_emoji} Status: {status_code}")
    logger.info(f"📍 Endpoint: {endpoint_name}")
    
    # Log response data with proper formatting
    if response_data:
        logger.info("📤 Response Data:")
        if isinstance(response_data, (dict, list)):
            # For large responses, use truncation
            max_length = 800 if status_code < 400 else 400
            formatted_json = format_json_data(response_data, max_length=max_length)
            for line in formatted_json.split('\n'):
                logger.info(f"   {line}")
        else:
            logger.info(f"   {response_data}")
    
    # Log final status
    if log_level == "error":
        logger.error(f"❌ {status_type} response from {endpoint_name}: Status {status_code}")
    else:
        logger.info(f"✅ {status_type} response from {endpoint_name}: Status {status_code}")
    
    log_separator(logger, "", "-", 60)

def log_database_operation(operation, table=None, success=True, error=None):
    """
    Log database operations in a structured format.
    
    Args:
        operation (str): Type of database operation (SELECT, INSERT, UPDATE, DELETE)
        table (str): Name of the table being operated on
        success (bool): Whether the operation was successful
        error (str): Error message if operation failed
    """
    logger = get_logger('database_logger')
    
    # Determine operation emoji and status
    operation_emojis = {
        'SELECT': '🔍',
        'INSERT': '➕',
        'UPDATE': '✏️',
        'DELETE': '🗑️',
        'CREATE': '🏗️',
        'DROP': '💥'
    }
    
    emoji = operation_emojis.get(operation.upper(), '🔧')
    status_emoji = "✅" if success else "❌"
    table_info = f" on table '{table}'" if table else ""
    
    if success:
        logger.info(f"{emoji} {status_emoji} Database {operation.upper()}{table_info} completed successfully")
    else:
        logger.error(f"{emoji} {status_emoji} Database {operation.upper()}{table_info} failed: {error}")

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