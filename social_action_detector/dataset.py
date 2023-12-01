import pandas as pd
import torch
from datasets import Dataset


def get_dataset(tokenizer):
    data = pd.read_csv('data/data.csv')
    descriptions = data['description'].tolist()
    print('Dataset size: ', len(descriptions))
    labels = data['label'].tolist()

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

    return train_dataset, val_dataset
