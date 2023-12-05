import os

from fastapi import FastAPI

from social_action_detector import config
from social_action_detector.main import load_model

app = FastAPI()
model, tokenizer = load_model(config.BERT_MODEL_NAME)

@app.get("/health")
async def health():
    return {"message": "Healthy"}

@app.get("/predict_bert")
async def predict(query : str):
    predict(query, model=model, tokenizer=tokenizer)

@app.get("/predict_llama2")
async def predict(query : str):
    from social_action_detector.handler import EndpointHandler
    handler = EndpointHandler()
    return handler({"inputs": query})
