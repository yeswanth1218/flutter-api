import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from prompts import SINGLE_IMAGE_PROMPT, MULTIPLE_IMAGES_PROMPT

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def process_image_with_gemini(image_data):
    """
    Process the image(s) with Gemini AI to extract business card information.
    
    Args:
        image_data: Either a single image bytes object or a list of image bytes objects
    
    Returns:
        dict: Response containing success status and extracted data or error message
    """
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Handle both single image and multiple images
        if isinstance(image_data, list):
            images = image_data
            num_images = len(images)
        else:
            images = [image_data]
            num_images = 1
        
        # Get the appropriate prompt based on number of images
        prompt = SINGLE_IMAGE_PROMPT if num_images == 1 else MULTIPLE_IMAGES_PROMPT
        
        # Prepare content for generation (prompt + images)
        content = [prompt] + images
        
        # Generate content using the image(s) and prompt
        response = model.generate_content(content)
        
        # Try to parse the response as JSON
        try:
            # Clean the response text to extract JSON
            response_text = response.text.strip()
            
            # Remove any markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            extracted_data = json.loads(response_text)
            
            # Validate and clean the extracted data
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
            
            return {
                "success": True,
                "data": cleaned_data
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse JSON response: {str(e)}",
                "raw_response": response.text
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Gemini AI processing error: {str(e)}"
        }