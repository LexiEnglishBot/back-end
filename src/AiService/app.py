import requests
from typing import Generator, Optional, List


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: Optional[str] = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    # === Text Generation ===
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        if stream:
            return self._stream_response(response)
        else:
            return response.json().get("response", "")

    def chat(self, messages: list[dict], stream: bool = False, **kwargs) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        if stream:
            return self._stream_response(response)
        else:
            return response.json().get("message", {}).get("content", "")

    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if line:
                l = line.decode("utf-8")
                print(l)
                yield l

    # === Model Management ===
    def pull_model(self, model_name: str) -> dict:
        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.ok

    def list_models(self) -> List[str]:
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models]

    def delete_model(self, model_name: str) -> dict:
        url = f"{self.base_url}/api/delete"
        payload = {"name": model_name}
        response = requests.delete(url, json=payload)
        response.raise_for_status()
        return response.json()


# === Example usage ===
if __name__ == "__main__":
    client = OllamaClient(model="llama3")

    # Pull a model
    print("Pulling model 'llama3'...")
    print(client.pull_model("llama3"))

    # List available models
    print("Available models:")
    print(client.list_models())

    # Use the model
    # print("Generating text:")
    # result = client.generate("Tell me a fun fact about space.")
    # print("The fun fact response:", result)

    # Chat example
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "What is quantum computing?"}
    # ]
    # chat_result = client.chat(messages)
    # print("Chat:", chat_result)

    text_content="""
In the wake of the terrorist attack on tourists at Baisaran, an off-the-road meadow, in South Kashmir’s Pahalgam Tuesday, in which at least 26 people were killed and several others injured, The Resistance Front (TRF), a shadow group of the banned Pakistan-based Lashkar-e-Taiba (LeT) terror group, claimed responsibility for the strike, according to sources in the central agencies.
"""

    prompt = f"""
In the long text I will write for you below, process the English words and provide an analysis of their difficulty or ease of learning for an average foreigner who is learning the language. No need for conjunctions such as from, to, to, etc.
The response format should also be in json format, with each word in its key and value having these values:
Difficulty level for learning from 1 to 10 - Meaning of the word in Persian - Example in a sentence - Meaning of the example in Persian - Learning in English - Related words up to 3 words in order of priority
Extract result for all words in the text and dont say anything else. Just write the json and return it.
Text:
{text_content}
"""
    result = client.generate(prompt)
    print("The response:", result)
