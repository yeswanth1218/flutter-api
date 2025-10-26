import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import modularized components
from routes import routes_bp
from logger_config import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logger.info("Flask application initialized")
logger.info("CORS enabled for all routes")

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

logger.info(f"Upload folder configured: {UPLOAD_FOLDER}")
logger.info(f"Max content length set to: {MAX_CONTENT_LENGTH} bytes")

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Uploads directory created/verified: {UPLOAD_FOLDER}")

# Register the routes blueprint
app.register_blueprint(routes_bp)
logger.info("Routes blueprint registered")

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    logger.warning(f"File too large error: {str(e)}")
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    logger.warning(f"404 error: {str(e)}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check if API key is configured
    if not os.getenv('GEMINI_API_KEY'):
        logger.warning("GEMINI_API_KEY not found in environment variables")
        logger.warning("Please create a .env file with your Gemini API key")
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your Gemini API key.")
    else:
        logger.info("GEMINI_API_KEY found and configured")
    
    logger.info("Starting Business Card Reader API on port 5001...")
    print("Starting Business Card Reader API on port 5001...")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise