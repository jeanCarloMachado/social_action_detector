import pandas as pd
import torch
from datasets import Dataset, load_dataset


def get_dataset(tokenizer):
    descriptions = load_train_data_from_hub()['train']['description']
    print('Dataset size: ', len(descriptions))
    labels = load_train_data_from_hub()['train']['label']

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


def load_train_data_from_hub():
    print("loading dataset from hub")
    dataset = load_dataset('JeanMachado/social_action_articles', data_files="data.csv")

    return dataset

def load_evaluation_data_from_hub():
    print("loading dataset from hub")
    dataset = load_dataset('JeanMachado/social_action_articles', data_files="holdout.csv")
    print ("Evaluation dataset size", len(dataset['train']))

    return dataset

if __name__ == "__main__":
    import fire
    fire.Fire()

