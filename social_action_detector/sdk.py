
import requests


def call_bert_api():
    requests.post("http://localhost:8000/predict/bert", json={"query": "a startup is creating a concept to turn poverty into history"})

if __file__ == "__main__":
    import fire
    fire.Fire()
