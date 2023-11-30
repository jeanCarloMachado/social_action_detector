
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
import torch.nn as nn
import config

def train(*, epochs: int = 10):

    if type(epochs) == str:
        epochs = int(epochs)
    print(" epochs: ", epochs)

    # Load the dataset from the CSV file
    data = pd.read_csv('data/data.csv')
    descriptions = data['description'].tolist()
    print('Dataset size: ', len(descriptions))
    labels = data['label'].tolist()

    # Tokenize the input data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(descriptions, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Prepare the dataset
    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']

    # Create a dictionary with the input data
    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    # Create the dataset
    dataset = Dataset.from_dict(data_dict)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    print("Valuation dataset size", len(val_dataset))

    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()
    trainer.model.save_pretrained(config.LOCAL_MODEL_NAME)
    tokenizer.save_pretrained(config.LOCAL_MODEL_NAME)
    print('Model trained and saved locally in the "results" folder!')

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
    model = load_model()
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
