from flask import Blueprint
from functions import (
    health_check,
    extract_business_card,
    register_user,
    login_user,
    get_user_cards,
    update_card_details,
    delete_or_restore,
    get_deleted_cards,
    add_category,
    get_categories
)
from logger_config import get_logger, log_request_info, log_response_info

# Initialize logger
logger = get_logger(__name__)

# Create Blueprint for routes
routes_bp = Blueprint('routes', __name__)
logger.info("Routes blueprint created")

@routes_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    from flask import request
    log_request_info(request, 'health_check')
    logger.info("Health check endpoint called")
    
    result = health_check()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'health_check')
    return result

@routes_bp.route('/extract-card', methods=['POST'])
def extract_card():
    """Extract business card information from uploaded image."""
    from flask import request
    log_request_info(request, 'extract_card')
    logger.info("Extract card endpoint called")
    
    result = extract_business_card()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'extract_card')
    return result

@routes_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    from flask import request
    log_request_info(request, 'register')
    logger.info("Register endpoint called")
    
    result = register_user()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'register')
    return result

@routes_bp.route('/login', methods=['POST'])
def login():
    """Authenticate user login."""
    from flask import request
    log_request_info(request, 'login')
    logger.info("Login endpoint called")
    
    result = login_user()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'login')
    return result

@routes_bp.route('/cards/<user_id>', methods=['GET'])
def cards(user_id):
    """Get all cards for a specific user."""
    from flask import request
    log_request_info(request, 'get_user_cards')
    logger.info(f"Get cards endpoint called for user: {user_id}")
    
    result = get_user_cards(user_id)
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'get_user_cards')
    return result

@routes_bp.route('/update_card_details', methods=['PUT'])
def update_card():
    """Update specific fields of a business card."""
    from flask import request
    log_request_info(request, 'update_card_details')
    logger.info("Update card details endpoint called")
    
    result = update_card_details()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'update_card_details')
    return result

@routes_bp.route('/delete_or_restore', methods=['PUT'])
def delete_restore():
    """Delete, restore, or deactivate a card based on the action parameter."""
    from flask import request
    log_request_info(request, 'delete_or_restore')
    logger.info("Delete or restore endpoint called")
    
    result = delete_or_restore()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'delete_or_restore')
    return result

@routes_bp.route('/deleted_cards', methods=['GET', 'POST'])
def deleted_cards():
    """Get all deleted cards for a specific user (status = 1)."""
    from flask import request
    log_request_info(request, 'get_deleted_cards')
    logger.info("Get deleted cards endpoint called")
    
    result = get_deleted_cards()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'get_deleted_cards')
    return result

@routes_bp.route('/add_category', methods=['POST'])
def add_cat():
    """Add a new category for a user."""
    from flask import request
    log_request_info(request, 'add_category')
    logger.info("Add category endpoint called")
    
    result = add_category()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'add_category')
    return result

@routes_bp.route('/get_categories', methods=['GET', 'POST'])
def get_cats():
    """Get all categories for a user with status = 0."""
    from flask import request
    log_request_info(request, 'get_categories')
    logger.info("Get categories endpoint called")
    
    result = get_categories()
    # Extract the actual response data for logging
    response_data = result[0] if isinstance(result[0], dict) else result[0].get_json() if hasattr(result[0], 'get_json') else str(result[0])
    log_response_info(response_data, result[1], 'get_categories')
    return result