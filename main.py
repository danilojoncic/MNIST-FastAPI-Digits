from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
from PIL import Image, ImageOps
import io
from starlette.templating import Jinja2Templates
from starlette.responses import HTMLResponse

from model_loader import MNISTPredictor

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = MNISTPredictor()


class ImagePayload(BaseModel):
    image: str  # base64 encoded string


@app.get("/paint-screen", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health-check")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_digit(payload: ImagePayload):
    try:
        # Remove header if present
        image_data = payload.image.split(",")[-1]
        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))

        image = Image.alpha_composite(white_bg, image)

        image = image.convert("L")

        if image.size != (28, 28):
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
        image_array = 255 - image_array
        image_array = image_array / 255.0
        print("Image shape:", image_array.shape)
        print("Image array (first 50 pixels):", image_array.flatten()[:50])
        print("Min pixel value:", image_array.min())
        print("Max pixel value:", image_array.max())
        print("Non-zero pixels:", np.count_nonzero(image_array))

        if len(image_array.shape) == 2:
            image_array = image_array.reshape(-1)

        result = predictor.predict(image_array)

        return {"prediction": result}

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/debug-image")
async def debug_image(payload: ImagePayload):
    try:
        image_data = payload.image.split(",")[-1]
        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white_bg, image)
        image = image.convert("L")

        if image.size != (28, 28):
            image = image.resize((28, 28), Image.Resampling.LANCZOS)

        image.save("debug_processed.png")

        image_array = np.array(image, dtype=np.float32)
        image_array = 255 - image_array
        image_array = image_array / 255.0

        return {
            "shape": image_array.shape,
            "min_value": float(image_array.min()),
            "max_value": float(image_array.max()),
            "non_zero_pixels": int(np.count_nonzero(image_array)),
            "mean_value": float(image_array.mean()),
            "saved_debug_image": "debug_processed.png"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))