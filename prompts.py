"""
LLM Prompts for Business Card Processing
This module contains all prompts used for processing business card images with AI models.
"""

# Single image business card extraction prompt
SINGLE_IMAGE_PROMPT = """
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

# Multiple images business card extraction prompt
MULTIPLE_IMAGES_PROMPT = """
You are an expert OCR (Optical Character Recognition) image-to-text extractor specializing in business card analysis. 
Your task is to carefully examine these business card images and extract all visible information with high accuracy.

Please analyze these business card images (front and back sides) and extract all the information in a structured JSON format. 
Combine information from both sides to create a comprehensive extraction.
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
7. When processing multiple images, combine all information from both sides into a single comprehensive JSON response.
"""