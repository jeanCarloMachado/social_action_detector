from typing import  Dict, List, Any
import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

class EndpointHandler():
    def __init__(self, path=""):
        # load the optimized model
        # load and use modepl
        print("init start")
        logging.info("test")


        self.path = path
        print("Login in")
        model_name  = path
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer
        print("init end")




    def __call__(self, data: Any) -> List[List[Dict[str, float]]]:
        """
        Args:
            data (:obj:):
                includes the input data and the parameters for the inference.
        Return:
            A :obj:`list`:. The object returned should be a list of one list like [[{"label": 0.9939950108528137}]] containing :
                - "label": A string representing what the label/class is. There can be multiple labels.
                - "score": A score between 0 and 1 describing how confident the model is for this label/class.
        """
        print("predict start")
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        tokenized_inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors='pt')
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        outputs = self.model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, dim=1)
        return predicted.item()


if __name__ == "__main__":
    import fire
    fire.Fire(EndpointHandler)