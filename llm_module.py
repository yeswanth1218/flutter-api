import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

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