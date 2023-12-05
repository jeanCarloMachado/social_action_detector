from social_action_detector import config
from social_action_detector.dataset import load_train_data_from_hub, load_evaluation_data_from_hub
from social_action_detector.main import predict, load_model


def compute(model_name = None):
    if not model_name:
        model_name = config.BERT_REMOTE_MODEL_NAME

    model, tokenizer = load_model(model_name)
    dataset = load_evaluation_data_from_hub()
    print(dataset)

    accurate = 0
    for row in dataset['train']:
        result = predict(row['description'], model=model, tokenizer=tokenizer)
        if result == row['label']:
            accurate += 1
        else:
            print("Wrong prediction: predicted {result}, actual: {row['label']}")


    print("Accuracy: ", accurate / len(dataset['train']))



if __name__ == "__main__":
    import fire
    fire.Fire()