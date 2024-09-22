import os
import requests
import json
from src.llm.prompt import get_prompt

# Go to https://www.awanllm.com/, create an account and get the free secret key
# remember to run in the command line < export AWAN_API_KEY="your-api-key" >
# or edit in the run Python configuration as environment variable
AWAN_API_KEY = os.getenv('AWAN_API_KEY')

url = "https://api.awanllm.com/v1/chat/completions"

prompt = get_prompt(
    context="Ciao a tutti, mi chiamo Riccardo Crupi e sono un data scientist in Intesa Sanpaolo",
    question="Chi è Riccardo? Come fa di cognome?")

payload = json.dumps({
  "model": "Awanllm-Llama-3-8B-Dolfin",
  "messages": [
    {
      "role": "user",
      "content": prompt
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.7
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {AWAN_API_KEY}"
}

# Remember to enable SSL verification in production!
response = requests.request("POST", url, headers=headers, data=payload, verify=False)

print(json.loads(response.text)['choices'][0]['message']['content'])

pass
