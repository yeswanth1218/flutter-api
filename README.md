# Business Card Reader API

A Flask-based API that uses Google's Gemini AI to extract information from business card images and return structured JSON data.

## Features

- Upload business card images via POST request
- Extract structured information using Gemini 2.0 Flash model
- Returns JSON formatted data with contact details, company info, and more
- Supports multiple image formats (PNG, JPG, JPEG, GIF, BMP, WebP)
- Built-in error handling and validation

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Add your API key to the `.env` file:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

3. **Run the application:**
   ```bash
   python app.py
   ```

The API will start on `http://localhost:5001`

## API Endpoints

### Health Check
- **GET** `/health`
- Returns API status

### Extract Business Card Information
- **POST** `/extract-card`
- Upload an image file with key `image`
- Returns extracted information in JSON format

## Example Usage

### Using curl:
```bash
curl -X POST -F "image=@business_card.jpg" http://localhost:6666/extract-card
```

### Example Response:
```json
{
  "success": true,
  "message": "Business card processed successfully",
  "data": {
    "name": "John Doe",
    "job_title": "Software Engineer",
    "company": "Tech Corp",
    "phone": "+1-555-123-4567",
    "email": "john.doe@techcorp.com",
    "website": "www.techcorp.com",
    "address": {
      "street": "123 Tech Street",
      "city": "San Francisco",
      "state": "CA",
      "zip_code": "94105",
      "country": "USA"
    },
    "social_media": {
      "linkedin": "linkedin.com/in/johndoe",
      "twitter": "@johndoe",
      "facebook": null,
      "instagram": null
    },
    "additional_info": "Mobile: +1-555-987-6543"
  }
}
```

## Error Handling

The API includes comprehensive error handling for:
- Missing or invalid files
- Unsupported file formats
- File size limits (16MB max)
- API key configuration issues
- Gemini AI processing errors

## File Structure

```
visit/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── .env              # Your actual environment variables (create this)
├── uploads/          # Directory for temporary file uploads (auto-created)
└── README.md         # This file
```