import os
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_with_gemini(image_data):
    """Process the image with Gemini AI to extract business card information."""
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Create the prompt for business card extraction
        prompt = """
        You are an expert OCR (Optical Character Recognition) image-to-text extractor specializing in business card analysis. 
        Your task is to carefully examine this business card image and extract all visible information with high accuracy.
        
        Please analyze this business card image and extract all the information in a structured JSON format. 
        Include the following fields if available:
        
        {
            "name": "Full name of the person",
            "job_title": "Job title or position",
            "company": "Company name",
            "phone": "Phone number(s)",
            "email": "Email address(es)",
            "website": "Website URL(s)",
            "address": {
                "street": "Street address",
                "city": "City",
                "state": "State/Province",
                "zip_code": "ZIP/Postal code",
                "country": "Country"
            },
            "social_media": {
                "linkedin": "LinkedIn profile",
                "twitter": "Twitter handle",
                "facebook": "Facebook profile",
                "instagram": "Instagram handle"
            },
            "additional_info": "Any other relevant information found on the card"
        }
        
        IMPORTANT INSTRUCTIONS:
        1. If any field is not available on the business card, set it to "None" (as a string).
        2. Be precise and accurate in text extraction.
        3. Maintain original formatting for phone numbers, emails, and URLs.
        4. Return only the JSON object, no additional text or formatting.
        5. Ensure the JSON is properly formatted and valid.
        """
        
        # Generate content using the image and prompt
        response = model.generate_content([prompt, image_data])
        
        # Try to parse the response as JSON
        try:
            # Clean the response text to extract JSON
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            extracted_data = json.loads(response_text)
            return extracted_data
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response.text
            }
            
    except Exception as e:
        return {
            "error": f"Error processing image with Gemini: {str(e)}"
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Business Card Reader API is running"})

@app.route('/extract-card', methods=['POST'])
def extract_business_card():
    """Extract information from a business card image."""
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        if file and allowed_file(file.filename):
            # Read the image file
            image_data = file.read()
            
            # Convert to PIL Image for processing
            try:
                image = Image.open(io.BytesIO(image_data))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process the image with Gemini AI
                extracted_info = process_image_with_gemini(image)
                
                return jsonify({
                    "success": True,
                    "message": "Business card processed successfully",
                    "data": extracted_info
                })
                
            except Exception as e:
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        else:
            return jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, webp"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Check if API key is configured
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your Gemini API key.")
    
    print("Starting Business Card Reader API on port 6666...")
    app.run(debug=True, host='0.0.0.0', port=5001)