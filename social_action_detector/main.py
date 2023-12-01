
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from social_action_detector import config
from social_action_detector.bert import train_bert
from social_action_detector.dataset import get_dataset
from social_action_detector.llama2 import train_llama

train = train_bert

def push_to_hub():
    model, tokenizer = load_model(config.LOCAL_MODEL_NAME)
    model.push_to_hub(config.MODEL_NAME)
    tokenizer.push_to_hub(config.MODEL_NAME)
    print('Model pushed to the hub!')


def push_data_to_hub():

    dataset = get_dataset()

def load_model(model_name):
    print("loading model: ", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def predict(description: str =  'a startup is creating a concept to turn poverty into history', model = None, tokenizer = None):
    print("predicting if it is a social good content")
    if model is None or tokenizer is None:
        model_name = config.LOCAL_MODEL_NAME
        print("loading model", model_name)
        model, tokenizer = load_model(model_name)


    tokenized_inputs = tokenizer(description, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    _, predicted = torch.max(outputs.logits, dim=1)
    return predicted.item()


if __name__ == "__main__":
    import fire
    fire.Fire()