import os
import json
import base64
import uuid
from datetime import datetime
from flask import request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io

from connections import get_db_connection
from llm_module import process_image_with_gemini, validate_business_card_images
from logger_config import get_logger, log_database_operation

# Initialize logger
logger = get_logger(__name__)
logger.info("Functions module initialized")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    logger.debug(f"Checking if file '{filename}' has allowed extension")
    result = '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    logger.debug(f"File '{filename}' allowed: {result}")
    return result

def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint called")
    response = {"status": "healthy", "message": "Business Card Reader API is running"}
    logger.info("Health check completed successfully")
    return jsonify(response), 200

def extract_business_card():
    """Extract business card information from uploaded image(s). Supports 1-2 images."""
    logger.info("Extract business card function called")
    
    try:
        # Check for images in request files
        images = []
        image_files = []
        
        logger.debug("Checking for image files in request")
        
        # Check for 'image' (single image) or 'image1', 'image2' (multiple images)
        if 'image' in request.files:
            # Single image case
            file = request.files['image']
            if file.filename != '':
                image_files.append(file)
                logger.info(f"Single image detected: {file.filename}")
        else:
            # Multiple images case - check for image1 and image2
            for i in range(1, 3):  # Support up to 2 images
                field_name = f'image{i}'
                if field_name in request.files:
                    file = request.files[field_name]
                    if file.filename != '':
                        image_files.append(file)
                        logger.info(f"Multiple image detected: {field_name} = {file.filename}")
        
        logger.info(f"Total images found: {len(image_files)}")
        
        # Validate we have at least one image
        if not image_files:
            logger.warning("No image files provided in request")
            return jsonify({"error": "No image file(s) provided. Use 'image' for single image or 'image1', 'image2' for multiple images"}), 400
        
        # Validate maximum 2 images
        if len(image_files) > 2:
            logger.warning(f"Too many images provided: {len(image_files)}")
            return jsonify({"error": "Maximum 2 images allowed"}), 400
        
        # Process each image file
        logger.info("Starting image processing and validation")
        for i, file in enumerate(image_files, 1):
            logger.debug(f"Processing image {i}: {file.filename}")
            
            # Check if file is allowed
            if not allowed_file(file.filename):
                logger.error(f"Invalid file type for {file.filename}")
                return jsonify({"error": f"Invalid file type for {file.filename}. Allowed types: png, jpg, jpeg, gif, bmp, webp"}), 400
            
            try:
                # Read the image file
                image_bytes = file.read()
                logger.debug(f"Read {len(image_bytes)} bytes from {file.filename}")
                
                # Validate that it's actually an image
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    image.verify()  # Verify that it's a valid image
                    logger.debug(f"Image validation successful for {file.filename}")
                except Exception as e:
                    logger.error(f"Image validation failed for {file.filename}: {str(e)}")
                    return jsonify({"error": f"Invalid image file: {file.filename}"}), 400
                
                # Reset file pointer and read again for processing
                file.seek(0)
                image_bytes = file.read()
                images.append(image_bytes)
                logger.debug(f"Image {file.filename} added to processing queue")
                
            except Exception as e:
                logger.error(f"Error processing image {file.filename}: {str(e)}")
                return jsonify({"error": f"Error processing image {file.filename}: {str(e)}"}), 400
        
        # Step 1: Validate if images contain business cards using AI
        logger.info("Starting AI validation of business card images")
        validation_result = validate_business_card_images(images)
        
        if not validation_result["success"]:
            logger.error(f"Image validation failed: {validation_result['error']}")
            return jsonify({"error": f"Image validation failed: {validation_result['error']}"}), 500
        
        # Check validation status
        validation_data = validation_result["validation"]
        logger.info(f"Validation status: {validation_data['status']}")
        
        if validation_data["status"] == "stop":
            logger.warning(f"Business card validation failed: {validation_data['reason']}")
            return jsonify({
                "success": False,
                "error": "Image validation failed",
                "reason": validation_data["reason"]
            }), 400
        
        # Step 2: Process the image(s) with Gemini AI for data extraction
        logger.info("Starting business card data extraction with Gemini AI")
        result = process_image_with_gemini(images)
        
        if not result["success"]:
            logger.error(f"Gemini AI processing failed: {result['error']}")
            return jsonify({"error": result["error"]}), 500
        
        extracted_data = result["data"]
        logger.info("Business card data extraction completed successfully")
        logger.debug(f"Extracted data keys: {list(extracted_data.keys())}")
        
        # Get user_id from request (if provided)
        user_id = request.form.get('user_id')
        logger.debug(f"User ID from request: {user_id}")
        
        # If user_id is provided, save to database
        if user_id:
            logger.info(f"Saving extracted data to database for user: {user_id}")
            try:
                # Validate UUID format
                uuid.UUID(user_id)
                logger.debug("User ID format validation successful")
                
                # Get database connection
                conn = get_db_connection()
                if not conn:
                    logger.error("Database connection failed")
                    return jsonify({"error": "Database connection failed"}), 500
                
                try:
                    cursor = conn.cursor()
                    
                    # Check if user exists
                    logger.debug("Checking if user exists in database")
                    cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
                    user_exists = cursor.fetchone()
                    
                    if not user_exists:
                        logger.warning(f"User not found in database: {user_id}")
                        return jsonify({"error": "User not found"}), 404
                    
                    logger.debug("User exists, proceeding with card insertion")
                    
                    # Generate unique card_id
                    card_id = str(uuid.uuid4())
                    logger.debug(f"Generated card ID: {card_id}")
                    
                    # Clean None values function
                    def clean_none_values(value):
                        if value is None or value == "None" or value == "":
                            return None
                        return value
                    
                    # Insert card data into database
                    logger.debug("Inserting card data into database")
                    insert_query = """
                        INSERT INTO cards (
                            card_id, user_id, name, job_title, company, phone, email, 
                            website, address, linkedin, twitter, facebook, instagram, 
                            additional_info, card_type, fav, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """
                    
                    cursor.execute(insert_query, (
                        card_id,
                        user_id,
                        clean_none_values(extracted_data.get("name")),
                        clean_none_values(extracted_data.get("job_title")),
                        clean_none_values(extracted_data.get("company")),
                        clean_none_values(extracted_data.get("phone")),
                        clean_none_values(extracted_data.get("email")),
                        clean_none_values(extracted_data.get("website")),
                        clean_none_values(extracted_data.get("address")),
                        clean_none_values(extracted_data.get("social_media", {}).get("linkedin")),
                        clean_none_values(extracted_data.get("social_media", {}).get("twitter")),
                        clean_none_values(extracted_data.get("social_media", {}).get("facebook")),
                        clean_none_values(extracted_data.get("social_media", {}).get("instagram")),
                        clean_none_values(extracted_data.get("additional_info")),
                        "business",  # card_type
                        0,  # fav (default value)
                        datetime.now()
                    ))
                    
                    log_database_operation("INSERT", "cards", True)
                    
                    # Commit the transaction
                    conn.commit()
                    logger.info(f"Business card data saved successfully with card_id: {card_id}")
                    
                    # Return success response with card_id
                    return jsonify({
                        "success": True,
                        "message": "Business card extracted and saved successfully",
                        "card_id": card_id,
                        "extracted_data": extracted_data
                    }), 200
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Database error during card insertion: {str(e)}")
                    log_database_operation("INSERT", "cards", False, str(e))
                    return jsonify({"error": f"Database error: {str(e)}"}), 500
                finally:
                    cursor.close()
                    conn.close()
                    logger.debug("Database connection closed")
                    
            except ValueError:
                logger.error(f"Invalid user_id format: {user_id}")
                return jsonify({"error": "Invalid user_id format"}), 400
            except Exception as e:
                logger.error(f"Database operation failed: {str(e)}")
                return jsonify({"error": f"Database operation failed: {str(e)}"}), 500
        else:
            # Return extracted data without saving to database
            logger.info("Returning extracted data without database save (no user_id provided)")
            return jsonify({
                "success": True,
                "message": "Business card extracted successfully",
                "extracted_data": extracted_data
            }), 200
                
    except Exception as e:
        logger.error(f"Server error in extract_business_card: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def register_user():
    """Register a new user."""
    logger.info("Register user function called")
    
    try:
        # Get JSON data from request
        data = request.get_json()
        logger.debug(f"Received registration data: {data is not None}")
        
        # Validate required fields
        if not data:
            logger.warning("No data provided in registration request")
            return jsonify({"error": "No data provided"}), 400
            
        name = data.get('user_name')
        phone = data.get('phone')
        password = data.get('password')
        
        logger.debug(f"Registration attempt for name: {name}, phone: {phone}")
        
        if not name or not phone or not password:
            logger.warning("Missing required fields in registration")
            return jsonify({"error": "user_name, phone, and password are required"}), 400
        
        # Basic validation
        if len(name.strip()) == 0:
            logger.warning("Empty name provided in registration")
            return jsonify({"error": "Name cannot be empty"}), 400
            
        if len(phone.strip()) == 0:
            logger.warning("Empty phone provided in registration")
            return jsonify({"error": "Phone cannot be empty"}), 400
            
        if len(password) < 6:
            logger.warning(f"Password too short: {len(password)} characters")
            return jsonify({"error": "Password must be at least 6 characters long"}), 400
        
        logger.info(f"Registration validation passed for phone: {phone}")
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed during registration")
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # Check if phone number already exists
            logger.debug("Checking if phone number already exists")
            cursor.execute("SELECT phone FROM users WHERE phone = %s", (phone,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                logger.warning(f"Phone number already registered: {phone}")
                return jsonify({"error": "Phone number already registered"}), 409
            
            logger.debug("Phone number is available, proceeding with registration")
            
            # Generate unique user_id
            user_id = str(uuid.uuid4())
            logger.debug(f"Generated user ID: {user_id}")
            
            # Encode password in base64
            encoded_password = base64.b64encode(password.encode()).decode()
            logger.debug("Password encoded successfully")
            
            # Insert new user
            logger.debug("Inserting new user into database")
            insert_query = """
                INSERT INTO users (user_id, name, phone, password, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                user_id,
                name.strip(),
                phone.strip(),
                encoded_password,
                'active',  # Default status
                datetime.now()
            ))
            
            log_database_operation("INSERT", "users", True)
            
            # Commit the transaction
            conn.commit()
            logger.info(f"User registered successfully with ID: {user_id}")
            
            return jsonify({
                "success": True,
                "message": "User registered successfully",
                "user_id": user_id,
                "name": name.strip(),
                "phone": phone.strip()
            }), 201
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error during user registration: {str(e)}")
            log_database_operation("INSERT", "users", False, str(e))
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            logger.debug("Database connection closed")
            
    except Exception as e:
        logger.error(f"Server error in register_user: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def login_user():
    """Authenticate user login."""
    logger.info("Login user function called")
    
    try:
        # Get JSON data from request
        data = request.get_json()
        logger.debug(f"Received login data: {data is not None}")
        
        # Validate required fields
        if not data:
            logger.warning("No data provided in login request")
            return jsonify({"error": "No data provided"}), 400
            
        phone = data.get('phone')
        password = data.get('password')
        
        logger.debug(f"Login attempt for phone: {phone}")
        
        if not phone or not password:
            logger.warning("Missing phone or password in login request")
            return jsonify({"error": "Phone and password are required"}), 400
        
        # Basic validation
        if len(phone.strip()) == 0:
            return jsonify({"error": "Phone cannot be empty"}), 400
            
        if len(password) == 0:
            return jsonify({"error": "Password cannot be empty"}), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # Check if user exists and get user data
            cursor.execute(
                "SELECT user_id, name, password, status FROM users WHERE phone = %s", 
                (phone.strip(),)
            )
            user_data = cursor.fetchone()
            
            if not user_data:
                return jsonify({"error": "Invalid phone number or password"}), 401
            
            user_id, name, stored_password, status = user_data
            
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

def toggle_favorite():
    """Toggle favorite status of a card (0 to 1 or 1 to 0)."""
    logger.info("Toggle favorite function called")
    
    try:
        # Get JSON data from request
        data = request.get_json()
        logger.debug(f"Received toggle favorite data: {data is not None}")
        
        # Validate required fields
        if not data:
            logger.warning("No data provided in toggle favorite request")
            return jsonify({"error": "No data provided"}), 400
            
        card_id = data.get('card_id')
        
        if not card_id:
            logger.warning("Missing card_id in toggle favorite request")
            return jsonify({"error": "card_id is required"}), 400
        
        # Validate UUID format for card_id
        try:
            uuid.UUID(card_id)
        except ValueError:
            logger.error(f"Invalid card_id format: {card_id}")
            return jsonify({"error": "Invalid card_id format"}), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed")
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            
            # First, check if the card exists and get current fav status
            cursor.execute("SELECT fav FROM cards WHERE card_id = %s", (card_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Card not found: {card_id}")
                return jsonify({"error": "Card not found"}), 404
            
            current_fav_status = result[0] if result[0] is not None else 0
            logger.debug(f"Current favorite status for card {card_id}: {current_fav_status}")
            
            # Toggle the favorite status (0 to 1, 1 to 0)
            new_fav_status = 1 if current_fav_status == 0 else 0
            
            # Update the favorite status
            cursor.execute(
                "UPDATE cards SET fav = %s WHERE card_id = %s",
                (new_fav_status, card_id)
            )
            
            if cursor.rowcount == 0:
                logger.error(f"Failed to update favorite status for card: {card_id}")
                return jsonify({"error": "Failed to update favorite status"}), 500
            
            log_database_operation("UPDATE", "cards", True)
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Favorite status toggled successfully for card {card_id}: {current_fav_status} -> {new_fav_status}")
            
            return jsonify({
                "success": True,
                "message": "Favorite status updated successfully",
                "card_id": card_id,
                "previous_status": current_fav_status,
                "new_status": new_fav_status
            }), 200
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error in toggle_favorite: {str(e)}")
            log_database_operation("UPDATE", "cards", False, str(e))
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            logger.debug("Database connection closed in toggle_favorite")
            
    except Exception as e:
        logger.error(f"Server error in toggle_favorite: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def get_user_cards(user_id):
    """Get all cards for a specific user."""
    try:
        logger.info(f"get_user_cards called with user_id: {user_id}")
        
        # Validate user_id format (should be UUID)
        try:
            uuid.UUID(user_id)
            logger.info(f"UUID validation successful for user_id: {user_id}")
        except ValueError as e:
            logger.error(f"UUID validation failed for user_id: {user_id}, error: {str(e)}")
            return jsonify({"error": "Invalid user_id format"}), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            logger.error("Database connection failed in get_user_cards")
            return jsonify({"error": "Database connection failed"}), 500
        
        logger.info("Database connection successful in get_user_cards")
        
        try:
            cursor = conn.cursor()
            
            # First, verify that the user exists
            logger.info(f"Checking if user exists: {user_id}")
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            user_exists = cursor.fetchone()
            
            if not user_exists:
                logger.warning(f"User not found in database: {user_id}")
                return jsonify({"error": "User not found"}), 404
            
            logger.info(f"User exists in database: {user_id}")
            
            # Get all active cards for the user (status = 0)
            logger.info(f"Fetching active cards for user: {user_id}")
            cursor.execute("""
                SELECT 
                    card_id, user_id, name, job_title, company, phone, email, 
                    website, address, linkedin, twitter, facebook, instagram, 
                    additional_info, tags, card_type, status, fav, created_at
                FROM cards 
                WHERE user_id = %s AND (status = 0 OR status IS NULL)
                ORDER BY created_at DESC
            """, (user_id,))
            
            cards_data = cursor.fetchall()
            logger.info(f"Found {len(cards_data)} cards for user: {user_id}")
            
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
                    "fav": card[17] if card[17] is not None else 0,
                    "created_at": card[18].isoformat() if card[18] else None
                }
                cards_list.append(card_dict)
            
            logger.info(f"Successfully formatted {len(cards_list)} cards for user: {user_id}")
            
            return jsonify({
                "success": True,
                "user_id": user_id,
                "cards": cards_list,
                "total_cards": len(cards_list)
            }), 200
            
        except Exception as e:
            logger.error(f"Database error in get_user_cards: {str(e)}")
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            logger.info("Database connection closed in get_user_cards")
            
    except Exception as e:
        logger.error(f"Server error in get_user_cards: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

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

def delete_or_restore():
    """Delete, restore, or deactivate a card based on the action parameter."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        card_id = data.get('card_id')
        action = data.get('action')
        
        if not card_id or not action:
            return jsonify({"error": "card_id and action are required"}), 400
        
        # Validate UUID format
        try:
            uuid.UUID(card_id)
        except ValueError:
            return jsonify({"error": "Invalid card_id format"}), 400
        
        # Validate action parameter
        valid_actions = ['inactive', 'delete', 'restore']
        if action not in valid_actions:
            return jsonify({"error": f"Invalid action. Must be one of: {', '.join(valid_actions)}"}), 400
        
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
            
            # Handle different actions
            if action == 'inactive':
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
                
                message = "Card marked as inactive successfully"
                
            elif action == 'delete':
                # Completely delete the card record from the database
                cursor.execute("DELETE FROM cards WHERE card_id = %s", (card_id,))
                
                # Check if any rows were affected
                if cursor.rowcount == 0:
                    return jsonify({"error": "Failed to delete card"}), 500
                
                message = "Card deleted permanently from database"
                
            elif action == 'restore':
                # Check if card is already active
                if current_status == 0:
                    return jsonify({"error": "Card is already active"}), 400
                
                # Update the card status from 1 (inactive) to 0 (active)
                cursor.execute("""
                    UPDATE cards 
                    SET status = 0, updated_at = %s
                    WHERE card_id = %s
                """, (datetime.now(), card_id))
                
                # Check if any rows were affected
                if cursor.rowcount == 0:
                    return jsonify({"error": "Failed to restore card"}), 500
                
                message = "Card restored successfully"
            
            # Commit the transaction
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": message,
                "card_id": card_id,
                "action": action
            }), 200
                
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def get_deleted_cards():
    """Get all deleted cards for a specific user (status = 1)."""
    try:
        user_id = None
        
        # Handle both GET and POST requests
        if request.method == 'GET':
            # For GET requests, try JSON body first (Flutter style), then query parameters
            try:
                if request.data:  # Check if there's actual data in the request body
                    data = request.get_json(force=True, silent=True)
                    if data and 'user_id' in data:
                        user_id = data.get('user_id')
            except Exception:
                pass
            
            # If no user_id from JSON body, try query parameters
            if not user_id:
                user_id = request.args.get('user_id')
        else:
            # For POST requests, get user_id from JSON data
            try:
                data = request.get_json()
                if data:
                    user_id = data.get('user_id')
            except Exception:
                # If JSON parsing fails, try query parameters as fallback
                user_id = request.args.get('user_id')
        
        # Check if user_id is provided
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Validate UUID format for user_id
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
            
            # Get all deleted cards for the user (status = 1)
            cursor.execute("""
                SELECT 
                    card_id, user_id, name, job_title, company, phone, email, 
                    website, address, linkedin, twitter, facebook, instagram, 
                    additional_info, tags, card_type, status, created_at
                FROM cards 
                WHERE user_id = %s AND status = 1
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
                    "status": card[16],
                    "created_at": card[17].isoformat() if card[17] else None
                }
                cards_list.append(card_dict)
            
            return jsonify({
                "success": True,
                "user_id": user_id,
                "deleted_cards": cards_list,
                "total_deleted_cards": len(cards_list)
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def add_category():
    """Add a new category for a user."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400,
        
        user_id = data.get('user_id')
        category_name = data.get('category_name')
        
        # Check if all required fields are provided
        if not user_id or not category_name:
            return jsonify({"error": "user_id and category_name are required"}), 400
        
        # Validate UUID format for user_id
        # try:
        #     uuid.UUID(user_id)
        # except ValueError:
        #     return jsonify({"error": "Invalid user_id format"}), 400
        
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

def get_categories():
    """Get all categories for a user with status = 0."""
    try:
        user_id = None
        
        # Handle both GET and POST requests
        if request.method == 'GET':
            # For GET requests, try JSON body first (Flutter style), then query parameters
            try:
                if request.data:  # Check if there's actual data in the request body
                    data = request.get_json(force=True, silent=True)
                    if data and 'user_id' in data:
                        user_id = data.get('user_id')
            except Exception:
                pass
            
            # If no user_id from JSON body, try query parameters
            if not user_id:
                user_id = request.args.get('user_id')
        else:
            # For POST requests, get user_id from JSON data
            try:
                data = request.get_json()
                if data:
                    user_id = data.get('user_id')
            except Exception:
                # If JSON parsing fails, try query parameters as fallback
                user_id = request.args.get('user_id')
        
        # Check if user_id is provided
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        # Validate UUID format for user_id
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
            
            # Check if user exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            user_exists = cursor.fetchone()
            
            if not user_exists:
                return jsonify({"error": "User not found"}), 404
            
            # Get all categories for the user with status = 0
            cursor.execute(
                "SELECT category_name FROM categories WHERE user_id = %s AND status = 0 ORDER BY category_name", 
                (user_id,)
            )
            categories = cursor.fetchall()
            
            # Extract category names from the result
            category_names = [category[0] for category in categories]
            
            return jsonify({
                "success": True,
                "user_id": user_id,
                "categories": category_names,
                "count": len(category_names)
            }), 200
                
        except Exception as e:
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500