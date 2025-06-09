import os
from copy import deepcopy
from typing import Tuple
import ollama
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import OpenAI
from datasets import load_dataset
from hdbscan import HDBSCAN
from pyarrow._dataset import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from torch import Tensor
from umap import UMAP

# Emotion labels
emotions_labels = [
    "happiness", "neutral", "sadness", "surprise", "love", "fear", "confusion",
    "disgust", "desire", "shame", "sarcasm", "anger", "guilt"
]

model = SentenceTransformer("thenlper/gte-small")
umap_model = UMAP(n_components=2, min_dist=0.0, metric='cosine')

hdbscan = HDBSCAN(
    min_cluster_size=50,
    metric="euclidean",
    cluster_selection_method="eom"
)
EMBEDDINGS_FILE = "embeddings.npy"
SENTENCES_FILE = "sentences.npy"

import requests

class Message:
    def __init__(self, content):
        self.content = content

class Choice:
    def __init__(self, message):
        self.message = message

class Response:
    def __init__(self, content):
        self.choices = [Choice(Message(content))]

class FakeOpenAIChatCompletions:
    def __init__(self, model="llama2", host="http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)

    def create(self, messages, **kwargs):
        chat_response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False
        )
        content = chat_response["message"]["content"]
        return Response(content)

class FakeOpenAIClient:
    def __init__(self, model="llama2", host="http://localhost:11434"):
        # Simulate openai.ChatCompletion.create(...) interface
        self.chat = type("Chat", (), {
            "completions": FakeOpenAIChatCompletions(model, host)
        })

def main():
    fake_client = FakeOpenAIClient()

    response = fake_client.chat.completions.create([
        {"role": "user", "content": "Give a short label for topic keywords: dogs, bark, walk, leash, pet"}
    ])

    print(response.choices[0].message.content)

    embeddings, sentences = build_sentence_embeddings()

    kmeans = KMeans(n_clusters=len(emotions_labels))

    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
        hdbscan_model=kmeans,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(sentences, embeddings)

    fig = topic_model.visualize_documents(sentences,
                                          topics,
                                          width=1200,
                                          hide_annotations=True,
                                          )
    fig.write_html("topic_viz.html")
    fig.show()
    original_topics = deepcopy(topic_model.get_topics())

    #    representation_model = MaximalMarginalRelevance(diversity=0.2)

    prompt = """I have a topic that contains the following sentences:
    [DOCUMENTS]
    
    The topic is described by the following keywords: '[KEYWORDS]'.
    
    Based on the following information above, extract a short topic label in the following format: 
    topic: <short topic label>"""
    fake_client = FakeOpenAIClient()

    representation_model = OpenAI(

        client=fake_client,
        model="minstral",
        exponential_backoff=True,
        chat=True,
        prompt=prompt
    )

    topic_model.update_topics(sentences, representation_model=representation_model)
    topic = topic_differences(topic_model, original_topics)
    topic.to_csv('output.csv')
    print(topic)


def topic_differences(model, original_topics):
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    topic_ids = list(original_topics.keys())

    for topic in topic_ids:
        og_words_raw = original_topics.get(topic)
        new_words_raw = model.get_topic(topic)

        if og_words_raw is None or new_words_raw is None:
            df.loc[len(df)] = [topic, "N/A", "N/A"]
            continue

        og_words = " _ ".join([w[0] for w in og_words_raw[:5]])
        new_words = " _ ".join([w[0] for w in new_words_raw[:5]])
        df.loc[len(df)] = [topic, og_words, new_words]

    return df


def build_sentence_embeddings() -> Tuple[Tensor, Dataset]:
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(SENTENCES_FILE):
        print("Loading cached embeddings...")
        sentences = np.load(SENTENCES_FILE, allow_pickle=True)
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        print("Computing embeddings from scratch...")
        dataset = load_dataset("boltuix/emotions-dataset", split="train")
        sentences = dataset["Sentence"]

        embeddings = model.encode(sentences, show_progress_bar=True)
        np.save(SENTENCES_FILE, np.array(sentences, dtype=object))  # dtype=object for variable-length strings
        np.save(EMBEDDINGS_FILE, embeddings)
    return embeddings, sentences


main()
