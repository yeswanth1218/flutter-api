"""
LLM Prompts for Business Card Processing
This module contains all prompts used for processing business card images with AI models.
"""

# Image validation prompt for contact card detection
IMAGE_VALIDATION_PROMPT = """
You are an expert image classifier specializing in identifying contact cards of all types.
Your task is to analyze the uploaded image(s) and determine if they contain any type of contact card or something completely different.

Please examine the image(s) carefully and classify them based on the following criteria:

ACCEPTABLE CONTACT CARDS (should PROCEED):
- BUSINESS CARDS: Professional cards with company info, job titles, business contact details
- PERSONAL CARDS: Personal contact cards with individual's name and contact information
- FREELANCE CARDS: Independent professional or freelancer contact cards
- PROFILE CARDS: Personal or professional profile cards with contact information
- NETWORKING CARDS: Any card designed for sharing contact information
- VISITING CARDS: Traditional visiting cards with personal/professional details

Key characteristics of acceptable cards:
- Contains contact information (name, phone, email, address, etc.)
- Card-like format (typically rectangular)
- Designed for sharing contact details
- Clear readable text with contact information
- May include logos, branding, or personal photos

UNACCEPTABLE CONTENT (should STOP):
- Random photos or images without contact information
- Screenshots of apps, websites, or documents
- ID cards, licenses, or official documents (not contact cards)
- Receipts, invoices, or financial documents
- Completely unrelated images (landscapes, objects, etc.)
- Unclear, blurry, or unreadable images
- Multiple unrelated items in the image

Based on your analysis, respond with a JSON object in the following format:

{
    "status": "proceed",
    "reason": "Image contains a business/contact card with readable information"
}

OR

{
    "status": "stop", 
    "reason": "Image does not contain a business/contact card - appears to be [specific description of what it actually is]"
}

IMPORTANT INSTRUCTIONS:
1. Use "proceed" status for ANY type of contact card (business, personal, freelance, profile, etc.)
2. Use "stop" status ONLY for non-contact-card content or unclear images
3. Focus on whether the image contains contact information in a card format
4. Provide a clear, specific reason for your decision
5. Return only the JSON object, no additional text
5. Ensure the JSON is properly formatted and valid
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