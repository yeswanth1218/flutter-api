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

# Create Blueprint for routes
routes_bp = Blueprint('routes', __name__)

@routes_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return health_check()

@routes_bp.route('/extract-card', methods=['POST'])
def extract_card():
    """Extract business card information from uploaded image."""
    return extract_business_card()

@routes_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    return register_user()

@routes_bp.route('/login', methods=['POST'])
def login():
    """Authenticate user login."""
    return login_user()

@routes_bp.route('/cards/<user_id>', methods=['GET'])
def cards(user_id):
    """Get all cards for a specific user."""
    return get_user_cards(user_id)

@routes_bp.route('/update_card_details', methods=['PUT'])
def update_card():
    """Update specific fields of a business card."""
    return update_card_details()

@routes_bp.route('/delete_or_restore', methods=['PUT'])
def delete_restore():
    """Delete, restore, or deactivate a card based on the action parameter."""
    return delete_or_restore()

@routes_bp.route('/deleted_cards', methods=['POST'])
def deleted_cards():
    """Get all deleted cards for a specific user (status = 1)."""
    return get_deleted_cards()

@routes_bp.route('/add_category', methods=['POST'])
def add_cat():
    """Add a new category for a user."""
    return add_category()

@routes_bp.route('/get_categories', methods=['POST'])
def get_cats():
    """Get all categories for a user with status = 0."""
    return get_categories()