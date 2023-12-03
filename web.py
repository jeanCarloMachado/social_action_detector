from fastapi import FastAPI
from handler import EndpointHandler

app = FastAPI()
handler = EndpointHandler()

@app.get("/health")
async def health():
    return {"message": "Hello World"}

@app.get("/predict")
async def predict(query : str):
    handler({"inputs": query})
