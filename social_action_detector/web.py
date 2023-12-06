import os

from fastapi import FastAPI

from social_action_detector import config
from social_action_detector.main import load_model, predict
from pydantic import BaseModel


app = FastAPI()
model, tokenizer = load_model(config.BERT_REMOTE_MODEL_NAME)

@app.get("/health")
async def health():
    return {"message": "Healthy"}


class Request(BaseModel):
    query: str

@app.post("/predict/bert")
async def predict_bert(request: Request):
    return predict(request.query, model=model, tokenizer=tokenizer)

@app.get("/predict/llama2")
async def predict_llama(query : str):
    from social_action_detector.handler import EndpointHandler
    handler = EndpointHandler()
    return handler({"inputs": query})
