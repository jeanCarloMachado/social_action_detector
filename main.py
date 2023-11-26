
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score

# Define the model
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.base_model = AutoModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[0])
        outputs = self.linear(outputs)
        return outputs

# Load the dataset from the CSV file
data = pd.read_csv('data.csv')
descriptions = data['description'].tolist()
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

# Define the model, optimizer, and loss function
model = BinaryClassification()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch'
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    pcompute_metrics=None,
    optimizers=(optimizer, None),
    data_collator=None,
    tokenizer=tokenizer
)

trainer.train()

