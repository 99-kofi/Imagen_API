from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import os
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

app = FastAPI()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/generate-image")
async def generate_image(
    image_file: UploadFile = File(...),
    prompt: str = "Add a red background to this image"
):
    try:
        # Verify image
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Process image
        image_data = await image_file.read()
        image = Image.open(BytesIO(image_data))
        
        # Generate with Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE']
            )
        )
        
        # Return generated image
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data'):
                img = Image.open(BytesIO(part.inline_data.data))
                img_bytes = BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                return StreamingResponse(img_bytes, media_type="image/png")
        
        raise HTTPException(400, "No image generated")
    
    except Exception as e:
        raise HTTPException(500, str(e)) 
