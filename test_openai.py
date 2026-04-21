import json
import requests

url = "http://127.0.0.1:11434/v1/chat/completions"
data = {
    "model": "smollm2",
    "messages": [
        {"role": "user", "content": "hola mundo"}
    ],
    "stream": True
}

try:
    response = requests.post(url, json=data, stream=True)
    print(f"Status: {response.status_code}")
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
