from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
import os
from google import genai
from google.genai import types

app = FastAPI()

# Initialize client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/generate-image")
async def generate_image(image_file: UploadFile = File(...)):
    try:
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        image = Image.open(BytesIO(await image_file.read()))
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=["Add a red background", image],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE']
            )
        )
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data'):
                img_bytes = BytesIO(part.inline_data.data)
                return StreamingResponse(img_bytes, media_type="image/png")
        
        raise HTTPException(400, "No image generated")
    
    except Exception as e:
        raise HTTPException(500, str(e))
