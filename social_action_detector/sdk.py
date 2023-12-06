
import requests

def call_bert_api(text_to_test='gerando falcoes is working to make the favelas a piece of museum'):
    result = requests.post("http://localhost:8080/predict/bert", json={"query": text_to_test})

    print(result.json())

if __name__ == "__main__":
    import fire
    fire.Fire()
