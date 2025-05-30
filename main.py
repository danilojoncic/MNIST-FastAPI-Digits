from fastapi import FastAPI

app = FastAPI()





@app.get("/health-check")
async def health_check():
    return {"status": "ok"}


@app.get("/model-details")
async def model_details():
    return {"model": "My Model"}

@app.post("/predict")
async def predict():
    return "Working on it!"