import os

from fastapi import FastAPI
from social_action_detector.handler import EndpointHandler

app = FastAPI()

if 'SKIP_HANDLER' not in os.environ:
    handler = EndpointHandler()
else:
    print("Skipping handler")

@app.get("/health")
async def health():
    return {"message": "Healthy"}

@app.get("/predict")
async def predict(query : str):
    return handler({"inputs": query})
