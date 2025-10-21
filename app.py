import os
import json
import base64
import uuid
import psycopg2
from datetime import datetime
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

def get_db_connection():
    """Get PostgreSQL database connection."""
    try:
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

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
            "phone": "Phone number(s) - extract only the numeric digits separated by commas (e.g., '9121697675, 7306515159')",
            "email": "Email address(es)",
            "website": "Website URL(s)",
            "address": "Complete address as it appears on the card",
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
        3. For phone numbers: Extract ONLY the numeric digits without country codes, parentheses, dashes, or spaces. If multiple phone numbers exist, separate them with commas and spaces (e.g., "9121697675, 7306515159").
        4. For emails and URLs: Maintain original formatting.
        5. Return only the JSON object, no additional text or formatting.
        6. Ensure the JSON is properly formatted and valid.
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
    """Extract information from a business card image and save to database."""
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        # Check if user_id is provided in form data
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Validate user_id format (should be UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            return jsonify({"error": "Invalid user_id format"}), 400
        
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
                
                # Check if extraction was successful
                if "error" in extracted_info:
                    return jsonify({
                        "success": False,
                        "error": "Failed to extract card information",
                        "details": extracted_info
                    }), 500
                
                # Generate unique card_id
                card_id = str(uuid.uuid4())
                
                # Get database connection
                conn = get_db_connection()
                if not conn:
                    return jsonify({"error": "Database connection failed"}), 500
                
                try:
                    cursor = conn.cursor()
                    
                    # Verify that the user exists
                    cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
                    user_exists = cursor.fetchone()
                    
                    if not user_exists:
                        return jsonify({"error": "User not found"}), 404
                    
                    # Prepare data for insertion
                    name = extracted_info.get('name', None)
                    job_title = extracted_info.get('job_title', None)
                    company = extracted_info.get('company', None)
                    phone = extracted_info.get('phone', None)
                    email = extracted_info.get('email', None)
                    website = extracted_info.get('website', None)
                    address = extracted_info.get('address', None)
                    
                    # Handle social media data
                    social_media = extracted_info.get('social_media', {})
                    linkedin = social_media.get('linkedin', None) if isinstance(social_media, dict) else None
                    twitter = social_media.get('twitter', None) if isinstance(social_media, dict) else None
                    facebook = social_media.get('facebook', None) if isinstance(social_media, dict) else None
                    instagram = social_media.get('instagram', None) if isinstance(social_media, dict) else None
                    
                    additional_info = extracted_info.get('additional_info', None)
                    
                    # Set default values
                    tags = []  # Empty array for now
                    card_type = 'business'  # Default card type
                    created_at = datetime.now()
                    
                    # Convert "None" strings to actual None values
                    def clean_none_values(value):
                        return None if value == "None" or value == "" else value
                    
                    name = clean_none_values(name)
                    job_title = clean_none_values(job_title)
                    company = clean_none_values(company)
                    phone = clean_none_values(phone)
                    email = clean_none_values(email)
                    website = clean_none_values(website)
                    address = clean_none_values(address)
                    linkedin = clean_none_values(linkedin)
                    twitter = clean_none_values(twitter)
                    facebook = clean_none_values(facebook)
                    instagram = clean_none_values(instagram)
                    additional_info = clean_none_values(additional_info)
                    
                    # Insert card data into database
                    insert_query = """
                        INSERT INTO cards (
                            card_id, user_id, name, job_title, company, phone, email, 
                            website, address, linkedin, twitter, facebook, instagram, 
                            additional_info, tags, card_type, status, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """
                    
                    cursor.execute(insert_query, (
                        card_id, user_id, name, job_title, company, phone, email,
                        website, address, linkedin, twitter, facebook, instagram,
                        additional_info, tags, card_type, 0, created_at
                    ))
                    
                    # Commit the transaction
                    conn.commit()
                    
                    return jsonify({
                        "success": True,
                        "message": "Business card processed and saved successfully",
                        "card_id": card_id,
                        "user_id": user_id,
                        "data": extracted_info
                    }), 201
                    
                except Exception as e:
                    conn.rollback()
                    return jsonify({"error": f"Database error: {str(e)}"}), 500
                finally:
                    cursor.close()
                    conn.close()
                
            except Exception as e:
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        else:
            return jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, webp"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/register', methods=['POST'])
def register_user():
    """Register a new user."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_name = data.get('user_name')
        phone = data.get('phone')
        password = data.get('password')
        email = data.get('email')  # Optional field
        
        # Check if all required fields are provided
        if not user_name or not phone or not password:
            return jsonify({"error": "user_name, phone, and password are required"}), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # Check if phone number already exists
            cursor.execute("SELECT phone FROM users WHERE phone = %s", (phone,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                return jsonify({"error": "Phone number already registered"}), 409
            
            # Generate unique user_id
            user_id = str(uuid.uuid4())
            
            # Encode password in base64
            encoded_password = base64.b64encode(password.encode()).decode()
            
            # Set default values
            account_type = 'free'
            status = 'active'
            created_at = datetime.now()
            
            # Handle email - if not provided, set to NULL (email column is now nullable)
            if not email:
                email = None
            
            # Insert new user into database (using 'name' column as per actual DB schema)
            insert_query = """
                INSERT INTO users (user_id, name, email, phone, password, account_type, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (user_id, user_name, email, phone, encoded_password, account_type, status, created_at))
            
            # Insert default categories for the new user
            default_categories = ['Business', 'Personal', 'Favorites', 'Important']
            category_insert_query = """
                INSERT INTO categories (user_id, category_name, status)
                VALUES (%s, %s, %s)
            """
            
            for category_name in default_categories:
                cursor.execute(category_insert_query, (user_id, category_name, 0))
            
            # Commit the transaction
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": "user created successfully",
                "user_id": user_id,
                "name": user_name
            }), 201
            
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/login', methods=['POST'])
def login_user():
    """Login user with phone and password."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        phone = data.get('phone')
        password = data.get('password')
        
        # Check if all required fields are provided
        if not phone or not password:
            return jsonify({"error": "phone and password are required"}), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # Check if phone number exists and get user details
            cursor.execute(
                "SELECT user_id, name, password, status FROM users WHERE phone = %s", 
                (phone,)
            )
            user_record = cursor.fetchone()
            
            if not user_record:
                return jsonify({"error": "Invalid phone number or password"}), 401
            
            user_id, name, stored_password, status = user_record
            
            # Check if user account is active
            if status != 'active':
                return jsonify({"error": "Account is not active"}), 401
            
            # Verify password (decode base64 stored password and compare)
            try:
                decoded_stored_password = base64.b64decode(stored_password).decode()
                if password != decoded_stored_password:
                    return jsonify({"error": "Invalid phone number or password"}), 401
            except Exception as e:
                return jsonify({"error": "Password verification failed"}), 500
            
            # Successful login
            return jsonify({
                "success": True,
                "message": "Login successful",
                "user_id": user_id,
                "name": name
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/cards/<user_id>', methods=['GET'])
def get_user_cards(user_id):
    """Get all cards for a specific user."""
    try:
        # Validate user_id format (should be UUID)
        try:
            uuid.UUID(user_id)
        except ValueError:
            return jsonify({"error": "Invalid user_id format"}), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # First, verify that the user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            user_exists = cursor.fetchone()
            
            if not user_exists:
                return jsonify({"error": "User not found"}), 404
            
            # Get all active cards for the user (status = 0)
            cursor.execute("""
                SELECT 
                    card_id, user_id, name, job_title, company, phone, email, 
                    website, address, linkedin, twitter, facebook, instagram, 
                    additional_info, tags, card_type, status, created_at
                FROM cards 
                WHERE user_id = %s AND (status = 0 OR status IS NULL)
                ORDER BY created_at DESC
            """, (user_id,))
            
            cards_data = cursor.fetchall()
            
            # Format the response
            cards_list = []
            for card in cards_data:
                card_dict = {
                    "card_id": str(card[0]),
                    "user_id": str(card[1]),
                    "name": card[2],
                    "job_title": card[3],
                    "company": card[4],
                    "phone": card[5],
                    "email": card[6],
                    "website": card[7],
                    "address": card[8],
                    "social_media": {
                        "linkedin": card[9],
                        "twitter": card[10],
                        "facebook": card[11],
                        "instagram": card[12]
                    },
                    "additional_info": card[13],
                    "tags": card[14] if card[14] else [],
                    "card_type": card[15],
                    "status": card[16] if card[16] is not None else 0,
                    "created_at": card[17].isoformat() if card[17] else None
                }
                cards_list.append(card_dict)
            
            return jsonify({
                "success": True,
                "user_id": user_id,
                "cards": cards_list,
                "total_cards": len(cards_list)
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/update_card_details', methods=['PUT'])
def update_card_details():
    """Update specific fields of a business card."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_id = data.get('user_id')
        card_id = data.get('card_id')
        updates = data.get('updates', {})
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
            
        if not card_id:
            return jsonify({"error": "card_id is required"}), 400
            
        if not updates or not isinstance(updates, dict):
            return jsonify({"error": "updates object is required and must be a dictionary"}), 400
            
        if len(updates) == 0:
            return jsonify({"error": "At least one field must be provided in updates"}), 400
        
        # Validate UUID formats
        try:
            uuid.UUID(user_id)
            uuid.UUID(card_id)
        except ValueError:
            return jsonify({"error": "Invalid user_id or card_id format"}), 400
        
        # Define allowed fields for update
        allowed_fields = {
            'name', 'job_title', 'company', 'phone', 'email', 'website', 
            'address', 'linkedin', 'twitter', 'facebook', 'instagram', 
            'additional_info', 'tags'
        }
        
        # Validate that only allowed fields are being updated
        invalid_fields = set(updates.keys()) - allowed_fields
        if invalid_fields:
            return jsonify({
                "error": f"Invalid fields: {', '.join(invalid_fields)}. Allowed fields: {', '.join(sorted(allowed_fields))}"
            }), 400
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            if not cursor.fetchone():
                return jsonify({"error": "User not found"}), 404
            
            # Check if card exists and belongs to the user
            cursor.execute(
                "SELECT card_id FROM cards WHERE card_id = %s AND user_id = %s", 
                (card_id, user_id)
            )
            if not cursor.fetchone():
                return jsonify({"error": "Card not found or does not belong to user"}), 404
            
            # Clean None values function
            def clean_none_values(value):
                if value is None or value == "None" or value == "":
                    return None
                return value
            
            # Build dynamic UPDATE query
            set_clauses = []
            values = []
            
            for field, value in updates.items():
                set_clauses.append(f"{field} = %s")
                values.append(clean_none_values(value))
            
            # Add updated timestamp
            set_clauses.append("updated_at = %s")
            values.append(datetime.now())
            
            # Add WHERE clause values
            values.extend([card_id, user_id])
            
            update_query = f"""
                UPDATE cards 
                SET {', '.join(set_clauses)}
                WHERE card_id = %s AND user_id = %s
            """
            
            cursor.execute(update_query, values)
            
            # Check if any rows were affected
            if cursor.rowcount == 0:
                return jsonify({"error": "No changes made to the card"}), 400
            
            # Commit the transaction
            conn.commit()
            
            # Fetch the updated card data
            cursor.execute("""
                SELECT card_id, user_id, name, job_title, company, phone, email, 
                       website, address, linkedin, twitter, facebook, instagram, 
                       additional_info, tags, card_type, status, created_at, updated_at
                FROM cards 
                WHERE card_id = %s AND user_id = %s
            """, (card_id, user_id))
            
            card_data = cursor.fetchone()
            
            if card_data:
                # Format the response
                updated_card = {
                    "card_id": card_data[0],
                    "user_id": card_data[1],
                    "name": card_data[2],
                    "job_title": card_data[3],
                    "company": card_data[4],
                    "phone": card_data[5],
                    "email": card_data[6],
                    "website": card_data[7],
                    "address": card_data[8],
                    "social_media": {
                        "linkedin": card_data[9],
                        "twitter": card_data[10],
                        "facebook": card_data[11],
                        "instagram": card_data[12]
                    },
                    "additional_info": card_data[13],
                    "tags": card_data[14],
                    "card_type": card_data[15],
                    "status": card_data[16] if card_data[16] is not None else 0,
                    "created_at": card_data[17].isoformat() if card_data[17] else None,
                    "updated_at": card_data[18].isoformat() if card_data[18] else None
                }
                
                return jsonify({
                    "success": True,
                    "message": "Card updated successfully",
                    "updated_fields": list(updates.keys()),
                    "card": updated_card
                }), 200
            else:
                return jsonify({"error": "Failed to retrieve updated card"}), 500
                
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/delete_card', methods=['POST'])
def delete_card():
    """Delete a card by updating its status from 0 (active) to 1 (inactive)."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        card_id = data.get('card_id')
        
        if not card_id:
            return jsonify({"error": "card_id is required"}), 400
        
        # Validate UUID format
        try:
            uuid.UUID(card_id)
        except ValueError:
            return jsonify({"error": "Invalid card_id format"}), 400
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # First, check if the status column exists in the cards table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'cards' AND column_name = 'status'
            """)
            status_column_exists = cursor.fetchone()
            
            # If status column doesn't exist, add it
            if not status_column_exists:
                cursor.execute("""
                    ALTER TABLE cards 
                    ADD COLUMN status INTEGER DEFAULT 0
                """)
                conn.commit()
                print("Status column added to cards table")
            
            # Check if card exists
            cursor.execute("SELECT card_id, status FROM cards WHERE card_id = %s", (card_id,))
            card_record = cursor.fetchone()
            
            if not card_record:
                return jsonify({"error": "Card not found"}), 404
            
            current_status = card_record[1] if len(card_record) > 1 else 0
            
            # Check if card is already inactive
            if current_status == 1:
                return jsonify({"error": "Card is already inactive"}), 400
            
            # Update the card status from 0 (active) to 1 (inactive)
            cursor.execute("""
                UPDATE cards 
                SET status = 1, updated_at = %s
                WHERE card_id = %s
            """, (datetime.now(), card_id))
            
            # Check if any rows were affected
            if cursor.rowcount == 0:
                return jsonify({"error": "Failed to update card status"}), 500
            
            # Commit the transaction
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": "Card deleted successfully (status updated to inactive)",
                "card_id": card_id
            }), 200
                
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/add_category', methods=['POST'])
def add_category():
    """Add a new category for a user."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_id = data.get('user_id')
        category_name = data.get('category_name')
        
        # Check if all required fields are provided
        if not user_id or not category_name:
            return jsonify({"error": "user_id and category_name are required"}), 400
        
        # Validate UUID format for user_id
        try:
            uuid.UUID(user_id)
        except ValueError:
            return jsonify({"error": "Invalid user_id format"}), 400
        
        # Validate category_name (basic validation)
        if not isinstance(category_name, str) or len(category_name.strip()) == 0:
            return jsonify({"error": "category_name must be a non-empty string"}), 400
        
        category_name = category_name.strip()
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            user_exists = cursor.fetchone()
            
            if not user_exists:
                return jsonify({"error": "User not found"}), 404
            
            # Check if category already exists for this user
            cursor.execute(
                "SELECT category_name, status FROM categories WHERE user_id = %s AND category_name = %s", 
                (user_id, category_name)
            )
            existing_category = cursor.fetchone()
            
            status_value = 0  # Default status for new categories
            
            if existing_category:
                # Category already exists, return the existing data with actual status
                existing_status = existing_category[1]
                return jsonify({
                    "success": True,
                    "message": "Category already exists",
                    "user_id": user_id,
                    "category_name": category_name,
                    "status": existing_status,
                    "already_exists": True
                }), 200
            else:
                # Insert new category
                insert_query = """
                    INSERT INTO categories (user_id, category_name, status)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(insert_query, (user_id, category_name, status_value))
                
                # Commit the transaction
                conn.commit()
                
                return jsonify({
                    "success": True,
                    "message": "Category added successfully",
                    "user_id": user_id,
                    "category_name": category_name,
                    "status": status_value,
                    "already_exists": False
                }), 201
                
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
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
    
    print("Starting Business Card Reader API on port 5001...")
    app.run(debug=True, host='0.0.0.0', port=5001)