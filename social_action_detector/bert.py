from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from social_action_detector import config
from social_action_detector.dataset import get_dataset

def train_bert(epochs = 1):
    if type(epochs) == str:
        epochs = int(epochs)
    print(" epochs: ", epochs)

    # Load the dataset from the CSV file

    # Tokenize the input data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset, val_dataset = get_dataset(tokenizer)


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