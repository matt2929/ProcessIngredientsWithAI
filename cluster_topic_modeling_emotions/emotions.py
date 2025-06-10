import os
from typing import Tuple

import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from bertopic.representation._base import BaseRepresentation
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from torch import Tensor
from umap import UMAP

# File paths
EMBEDDINGS_FILE = "embeddings.npy"
SENTENCES_FILE = "sentences.npy"
TOPICS_FILE = "topics.npy"
PROBS_FILE = "probs.npy"
TOPIC_MODEL_FILE = "topic_model"

# Models
embedding_model = SentenceTransformer("thenlper/gte-small")
umap_model = UMAP(n_components=3, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom")


class GeminiRepresentation(BaseRepresentation):
    def __init__(self,api_key, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def __call__(self, documents, ctfidf, words, embeddings=None):
        prompt = f"""
        I have a topic that contains the following sentences:
        {documents}

        The topic is described by the following keywords: '{words}'.

        Based on the following information above, respond with a short topic label:
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()


def main():
    # Load or build embeddings
    embeddings, sentences = load_or_build_embeddings()

    # Load or fit model
    if os.path.exists(TOPICS_FILE) and os.path.exists(PROBS_FILE) and os.path.exists(TOPIC_MODEL_FILE):
        print("Loading cached topic model and results...")
        topic_assignment = np.load(TOPICS_FILE, allow_pickle=True)
        probs = np.load(PROBS_FILE, allow_pickle=True)
        topic_model = BERTopic.load(TOPIC_MODEL_FILE)
    else:
        print("Fitting new BERTopic model...")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True
        )
        topic_assignment, probs = topic_model.fit_transform(sentences, embeddings)
        np.save(TOPICS_FILE, topic_assignment)
        np.save(PROBS_FILE, probs)
        topic_model.save(TOPIC_MODEL_FILE)

    # Use Gemini to generate topic names
    api_key = os.getenv("GEMINI_API_KEY")  # Or hardcode it here for testing
    representation_model = GeminiRepresentation(api_key=api_key)
    print(representation_model.model.generate_content("test. please return something"))
    topic_model.update_topics(sentences, representation_model=representation_model)
    topics = topic_model.get_topics()

    # Overwrite topic labels
    new_labels = {}
    for topic_num in topics:
        if topic_num == -1:
            continue  # Skip outliers
        words = topic_model.get_topic(topic_num)
        docs = topic_model.get_representative_docs(topic_num)
        doc_texts = [sentences[int(i)] for i in docs if isinstance(i, (int, float, str)) and str(i).isdigit()]
        label = representation_model(doc_texts, None, [w for w, _ in words])
        new_labels[topic_num] = [(label, 1.0)]

    topic_model.topic_representations_ = new_labels

    # Visualize
    visualize_3d_topics(sentences, embeddings, topic_assignment, topic_model)


def load_or_build_embeddings() -> Tuple[Tensor, list]:
    """Load cached embeddings or compute them from scratch."""
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(SENTENCES_FILE):
        print("Loading cached embeddings...")
        paragraphs = np.load(SENTENCES_FILE, allow_pickle=True)
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        print("Computing embeddings from scratch...")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
        raw_lines = dataset["text"]
        text = "\n".join(raw_lines)
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

        print(f"Using {len(paragraphs)} grouped paragraphs.")
        embeddings = embedding_model.encode(paragraphs, show_progress_bar=True)
        np.save(SENTENCES_FILE, np.array(paragraphs, dtype=object))
        np.save(EMBEDDINGS_FILE, embeddings)
    return embeddings, paragraphs


def visualize_3d_topics(sentences, embeddings, topic_assignment, topic_model):
    """Reduce to 3D and visualize topics using Plotly."""
    reduced_embeddings = umap_model.fit_transform(embeddings)
    topic_words = topic_model.get_topics()
    topic_labels = [
        topic_words[t][0][0] if t in topic_words and topic_words[t] else "Outlier"
        for t in topic_assignment
    ]
    trim_size = 100
    short_texts = [s[:trim_size] + "..." if len(s) > trim_size else s for s in sentences]

    # Print preview
    label_previews = {}
    for label, snippet in zip(topic_labels, short_texts):
        label_previews.setdefault(label, [])
        if len(label_previews[label]) < 3:
            label_previews[label].append(snippet)

    for label, samples in label_previews.items():
        print(label)
        for s in samples:
            print(f"\t- '{s}'")
        print("=" * 10)

    # Plot
    df_plot = pd.DataFrame(reduced_embeddings, columns=["x", "y", "z"])
    df_plot["topic"] = topic_labels
    df_plot["text"] = short_texts
    if len(df_plot) > 3000:
        print(f"⚠️ Warning: {len(df_plot)} points may slow down visualization")

    fig = px.scatter_3d(df_plot, x="x", y="y", z="z",
                        color="topic",
                        title="3D Topic Visualization")
    fig.update_traces(marker=dict(size=3))
    fig.write_html("topic_viz_light.html", auto_open=False)
    fig.show()


if __name__ == "__main__":
    main()
