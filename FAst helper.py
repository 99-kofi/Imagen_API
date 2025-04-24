from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="Gemini Image Generator API",
             description="API for generating modified images using Google Gemini",
             version="1.0.0")

class ImageRequest(BaseModel):
    prompt: str = "Add a red background to this image and generate it"
    model: Optional[str] = "gemini-2.0-flash-exp-image-generation"

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=GEMINI_API_KEY)

@app.post("/generate-image")
async def generate_image(
    prompt: str = "Add a red background to this image and generate it",
    image_file: UploadFile = File(...),
    model: Optional[str] = "gemini-2.0-flash-exp-image-generation"
):
    try:
        # Validate file type
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read the uploaded image (limit to 5MB)
        max_size = 5 * 1024 * 1024  # 5MB
        image_data = await image_file.read()
        if len(image_data) > max_size:
            raise HTTPException(status_code=413, detail="Image too large (max 5MB)")
        
        image = Image.open(BytesIO(image_data))

        # Generate content using Gemini
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        # Process response
        if not response.candidates:
            return JSONResponse({"error": "No response generated"}, status_code=400)

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data'):
                generated_image = Image.open(BytesIO(part.inline_data.data))
                img_byte_arr = BytesIO()
                generated_image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return StreamingResponse(img_byte_arr, media_type="image/png")

        return JSONResponse({"error": "No image generated"}, status_code=400)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)