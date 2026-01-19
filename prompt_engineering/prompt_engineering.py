import json
from typing import List, Dict
from rich.console import Console
from rich.markdown import Markdown

import requests

# --- Ollama API endpoint ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"

user_role = "user"
system_role = "system"
assistant_role = "assistant"


# --- Your conversation prompt ---
def build_prompt(root: List[Dict[str, str]], turn: bool) -> List[Dict[str, str]]:
    if turn:
        content = (
            "You are performing an interview for a Senior Software Engineer Job. "
            "You are responsible for a System design question. Ask a question that "
            "the candidate should take about 30 minutes to answer. Be sure to push "
            "the candidate to do some back-of-the-napkin math, discuss trade-offs, "
            "and explain why they would choose certain systems. "
            "You don't need to ask everything at once — you will be able to probe follow-up questions."
            "Its incredibly important that you hire a great candidate be sure to probe and gain confidence this is "
            "The optimal hire. Your failure to do so could result in the loss of your job."
        )
    else:
        content = (
            "You are a proficient and experienced Software developer interviewing "
            "for a job as a Senior Software Engineer. You are in a 30-minute system "
            "design interview. You are tasked to design a system. Be sure to provide "
            "some initial data, discuss trade-offs, and deeply explain your reasoning. "
            "You don't need to ask everything at once — you will be asked follow-up questions."
        )

    system_start = [{"content": content, "role": system_role}]

    alt_index = 0  # separate index for alternation
    for prompt in root:
        if prompt["role"] == system_role:
            continue

        # Adjust alternation based on whose turn it is
        actual_index = alt_index + (1 if turn else 0)

        if actual_index % 2 != 0:
            system_start.append({"content": prompt["content"], "role": user_role})
        else:
            system_start.append({"content": prompt["content"], "role": assistant_role})

        alt_index += 1  # only increment when we keep the prompt

    return system_start

console = Console()
# --- Conversation loop ---
chain = [{'role': user_role, 'content': 'Hello'}]

for i in range(10):
    chain = build_prompt(chain, i % 2 == 0)
    data = {
        "model": "llama3",  # Or your Ollama model
        "messages": chain,
        "stream": False
    }

    response = requests.post(OLLAMA_API_URL, json=data)
    response_json = response.json()

    message = response_json.get('message', {}).get('content')
    if message:
        chain.append(response_json.get('message', {}))

    output = chain[-1]['content'].replace('\\n', '\n')
    console.print(Markdown(f"{'Candidate' if i % 2 else 'Interviewer'}\n{'-'*10}\n{output}"))
print(message)
