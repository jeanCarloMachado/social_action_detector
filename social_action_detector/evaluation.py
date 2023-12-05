from social_action_detector import config
from social_action_detector.dataset import , load_evaluation_data_from_hub


def compute_gpt4():
    from social_action_detector.gpt4 import predict
    compute(model_name="gpt-4", predict_function=predict)



def compute(model_name = None, predict_function=None):
    dataset = load_evaluation_data_from_hub()
    print(dataset)


    if not predict_function:
        from social_action_detector.main import predict
        if not model_name:
            model_name = config.BERT_REMOTE_MODEL_NAME

        predict_function = lambda description: predict(description, model_name=model_name)



    accurate = 0
    for row in dataset['train']:
        result = predict_function(row['description'])
        if result == row['label']:
            accurate += 1
        else:
            print(f"Wrong prediction: predicted {result}, actual: {row['label']}")


    print("Final Accuracy: ", accurate / len(dataset['train']))



if __name__ == "__main__":
    import fire
    fire.Fire()