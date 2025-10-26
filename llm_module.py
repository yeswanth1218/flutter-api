import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
from prompts import SINGLE_IMAGE_PROMPT, MULTIPLE_IMAGES_PROMPT, IMAGE_VALIDATION_PROMPT
from logger_config import get_logger, log_llm_operation

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
logger.info("Gemini AI configured successfully")

def process_image_with_gemini(image_data):
    """
    Process the image(s) with Gemini AI to extract business card information.
    
    Args:
        image_data: Either a single image bytes object or a list of image bytes objects
    
    Returns:
        dict: Response containing success status and extracted data or error message
    """
    start_time = time.time()
    logger.info("Starting image processing with Gemini AI")
    
    try:
        # Initialize the Gemini model using environment variable
        model_name = os.getenv('LLM_MODEL', 'gemini-2.5-flash-lite')  # Default fallback
        logger.info(f"Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Handle both single image and multiple images
        if isinstance(image_data, list):
            raw_images = image_data
            logger.info(f"Processing {len(raw_images)} images")
        else:
            raw_images = [image_data]
            logger.info("Processing single image")
        
        # Convert bytes to proper format for Gemini
        images = []
        for i, img_bytes in enumerate(raw_images):
            logger.debug(f"Converting image {i+1} to Gemini format")
            # Create a proper image object that Gemini can understand
            image_part = {
                "mime_type": "image/jpeg",  # Default to JPEG, Gemini will handle other formats
                "data": img_bytes
            }
            images.append(image_part)
        
        # Get the appropriate prompt based on number of images
        prompt = SINGLE_IMAGE_PROMPT if len(raw_images) == 1 else MULTIPLE_IMAGES_PROMPT
        logger.info(f"Using {'single' if len(raw_images) == 1 else 'multiple'} image prompt")
        
        # Prepare content for generation (prompt + images)
        content = [prompt] + images
        
        # Generate content using the image(s) and prompt
        logger.info("Sending request to Gemini AI for content generation")
        response = model.generate_content(content)
        logger.info("Received response from Gemini AI")
        
        # Try to parse the response as JSON
        try:
            logger.debug("Attempting to parse Gemini response as JSON")
            # Clean the response text to extract JSON
            response_text = response.text.strip()
            logger.debug(f"Raw response length: {len(response_text)} characters")
            
            # Remove any markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                logger.debug("Removed ```json prefix")
            if response_text.startswith('```'):
                response_text = response_text[3:]
                logger.debug("Removed ``` prefix")
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                logger.debug("Removed ``` suffix")
            
            response_text = response_text.strip()
            
            # Parse JSON
            extracted_data = json.loads(response_text)
            logger.info("Successfully parsed JSON response from Gemini")
            
            # Validate and clean the extracted data
            logger.debug("Cleaning and validating extracted data")
            cleaned_data = {
                "name": extracted_data.get("name", "None"),
                "job_title": extracted_data.get("job_title", "None"),
                "company": extracted_data.get("company", "None"),
                "phone": extracted_data.get("phone", "None"),
                "email": extracted_data.get("email", "None"),
                "website": extracted_data.get("website", "None"),
                "address": extracted_data.get("address", "None"),
                "social_media": {
                    "linkedin": extracted_data.get("social_media", {}).get("linkedin", "None"),
                    "twitter": extracted_data.get("social_media", {}).get("twitter", "None"),
                    "facebook": extracted_data.get("social_media", {}).get("facebook", "None"),
                    "instagram": extracted_data.get("social_media", {}).get("instagram", "None")
                },
                "additional_info": extracted_data.get("additional_info", "None")
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Image processing completed successfully in {processing_time:.2f} seconds")
            log_llm_operation("image_processing", model_name, True, None, processing_time)
            
            return {
                "success": True,
                "data": cleaned_data
            }
            
        except json.JSONDecodeError as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to parse JSON response: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Raw response that failed to parse: {response.text}")
            log_llm_operation("image_processing", model_name, False, error_msg, processing_time)
            
            return {
                "success": False,
                "error": error_msg,
                "raw_response": response.text
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Gemini AI processing error: {str(e)}"
        logger.error(error_msg)
        log_llm_operation("image_processing", model_name, False, error_msg, processing_time)
        
        return {
            "success": False,
            "error": error_msg
        }

def validate_business_card_images(image_data):
    """
    Validate if the uploaded image(s) contain business cards using Gemini AI.
    
    Args:
        image_data: Either a single image bytes object or a list of image bytes objects
    
    Returns:
        dict: Response containing validation result with status and reason
    """
    start_time = time.time()
    logger.info("Starting business card image validation")
    
    try:
        # Initialize the Gemini model using environment variable
        model_name = os.getenv('LLM_MODEL', 'gemini-2.0-flash-exp')  # Default fallback
        logger.info(f"Using model for validation: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Handle both single image and multiple images
        if isinstance(image_data, list):
            raw_images = image_data
            logger.info(f"Validating {len(raw_images)} images")
        else:
            raw_images = [image_data]
            logger.info("Validating single image")
        
        # Convert bytes to proper format for Gemini
        images = []
        for i, img_bytes in enumerate(raw_images):
            logger.debug(f"Converting validation image {i+1} to Gemini format")
            # Create a proper image object that Gemini can understand
            image_part = {
                "mime_type": "image/jpeg",  # Default to JPEG, Gemini will handle other formats
                "data": img_bytes
            }
            images.append(image_part)
        
        # Prepare content for generation (validation prompt + images)
        content = [IMAGE_VALIDATION_PROMPT] + images
        
        # Generate content using the image(s) and validation prompt
        logger.info("Sending validation request to Gemini AI")
        response = model.generate_content(content)
        logger.info("Received validation response from Gemini AI")
        
        # Try to parse the response as JSON
        try:
            logger.debug("Attempting to parse validation response as JSON")
            # Clean the response text to extract JSON
            response_text = response.text.strip()
            logger.debug(f"Validation response length: {len(response_text)} characters")
            
            # Remove any markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                logger.debug("Removed ```json prefix from validation response")
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                logger.debug("Removed ``` suffix from validation response")
            
            # Parse JSON response
            validation_result = json.loads(response_text.strip())
            logger.info("Successfully parsed validation JSON response")
            
            # Validate the required fields are present
            if "status" not in validation_result or "reason" not in validation_result:
                error_msg = "Invalid validation response format - missing required fields"
                logger.error(error_msg)
                processing_time = time.time() - start_time
                log_llm_operation("image_validation", model_name, False, error_msg, processing_time)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "raw_response": response.text
                }
            
            # Validate status field has correct values
            if validation_result["status"] not in ["proceed", "stop"]:
                error_msg = "Invalid status value - must be 'proceed' or 'stop'"
                logger.error(error_msg)
                processing_time = time.time() - start_time
                log_llm_operation("image_validation", model_name, False, error_msg, processing_time)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "raw_response": response.text
                }
            
            processing_time = time.time() - start_time
            logger.info(f"Image validation completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Validation result: {validation_result['status']} - {validation_result['reason']}")
            log_llm_operation("image_validation", model_name, True, None, processing_time)
            
            return {
                "success": True,
                "validation": validation_result
            }
            
        except json.JSONDecodeError as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to parse validation JSON response: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Raw validation response that failed to parse: {response.text}")
            log_llm_operation("image_validation", model_name, False, error_msg, processing_time)
            
            return {
                "success": False,
                "error": error_msg,
                "raw_response": response.text
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Validation error: {str(e)}"
        logger.error(error_msg)
        log_llm_operation("image_validation", model_name, False, error_msg, processing_time)
        
        return {
            "success": False,
            "error": error_msg
        }