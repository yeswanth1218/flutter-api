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
from llm_module import process_image_with_gemini

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Business Card Reader API is running"}), 200

def extract_business_card():
    """Extract business card information from uploaded image."""
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, webp"}), 400
        
        try:
            # Read the image file
            image_bytes = file.read()
            
            # Validate that it's actually an image
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.verify()  # Verify that it's a valid image
            except Exception as e:
                return jsonify({"error": "Invalid image file"}), 400
            
            # Reset file pointer and read again for processing
            file.seek(0)
            image_bytes = file.read()
            
            # Process the image with Gemini AI
            result = process_image_with_gemini(image_bytes)
            
            if not result["success"]:
                return jsonify({"error": result["error"]}), 500
            
            extracted_data = result["data"]
            
            # Get user_id from request (if provided)
            user_id = request.form.get('user_id')
            
            # If user_id is provided, save to database
            if user_id:
                try:
                    # Validate UUID format
                    uuid.UUID(user_id)
                    
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
                        
                        # Generate unique card_id
                        card_id = str(uuid.uuid4())
                        
                        # Clean None values function
                        def clean_none_values(value):
                            if value is None or value == "None" or value == "":
                                return None
                            return value
                        
                        # Insert card data into database
                        insert_query = """
                            INSERT INTO cards (
                                card_id, user_id, name, job_title, company, phone, email, 
                                website, address, linkedin, twitter, facebook, instagram, 
                                additional_info, card_type, created_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                            "business_card",  # card_type
                            datetime.now()
                        ))
                        
                        # Commit the transaction
                        conn.commit()
                        
                        # Return success response with card_id
                        return jsonify({
                            "success": True,
                            "message": "Business card extracted and saved successfully",
                            "card_id": card_id,
                            "extracted_data": extracted_data
                        }), 200
                        
                    except Exception as e:
                        conn.rollback()
                        return jsonify({"error": f"Database error: {str(e)}"}), 500
                    finally:
                        cursor.close()
                        conn.close()
                        
                except ValueError:
                    return jsonify({"error": "Invalid user_id format"}), 400
                except Exception as e:
                    return jsonify({"error": f"Database operation failed: {str(e)}"}), 500
            else:
                # Return extracted data without saving to database
                return jsonify({
                    "success": True,
                    "message": "Business card extracted successfully",
                    "extracted_data": extracted_data
                }), 200
                
        except Exception as e:
            return jsonify({"error": f"Image processing error: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def register_user():
    """Register a new user."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        name = data.get('name')
        phone = data.get('phone')
        password = data.get('password')
        
        if not name or not phone or not password:
            return jsonify({"error": "Name, phone, and password are required"}), 400
        
        # Basic validation
        if len(name.strip()) == 0:
            return jsonify({"error": "Name cannot be empty"}), 400
            
        if len(phone.strip()) == 0:
            return jsonify({"error": "Phone cannot be empty"}), 400
            
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters long"}), 400
        
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
            
            # Insert new user
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
            
            # Commit the transaction
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": "User registered successfully",
                "user_id": user_id,
                "name": name.strip(),
                "phone": phone.strip()
            }), 201
            
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def login_user():
    """Authenticate user login."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        phone = data.get('phone')
        password = data.get('password')
        
        if not phone or not password:
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
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_id = data.get('user_id')
        
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

def get_categories():
    """Get all categories for a user with status = 0."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        user_id = data.get('user_id')
        
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