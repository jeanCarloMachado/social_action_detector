
import torch
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config
from bert import train_bert
from llama2 import train_llama

train = train_bert

def push_to_hub():
    model, tokenizer = load_model()
    model.push_to_hub(config.MODEL_NAME)
    tokenizer.push_to_hub(config.MODEL_NAME)
    print('Model pushed to the hub!')


def load_model(use_local=True):
    model_name = config.LOCAL_MODEL_NAME if use_local else config.FULL_MODEL_NAME
    print("loading model: ", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def predict(description: str =  'a startup is creating a concept to turn poverty into history', use_local=True):
    model_name = config.LOCAL_MODEL_NAME if use_local else config.FULL_MODEL_NAME
    model, tokenizer = load_model()
    print("predicting if it is a social good content")
    tokenized_inputs = tokenizer(description, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    _, predicted = torch.max(outputs.logits, dim=1)
    return predicted.item()


if __name__ == "__main__":
    import fire
    fire.Fire()