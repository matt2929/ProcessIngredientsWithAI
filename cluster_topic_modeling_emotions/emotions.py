import os
from typing import Tuple, Mapping, List

import google.generativeai as genai
import numpy as np
import pandas as pd
import plotly.express as px
from bertopic import BERTopic
from tqdm import tqdm
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters

from bertopic.representation._base import BaseRepresentation
from bertopic.representation import KeyBERTInspired
from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from scipy.sparse import csr_matrix
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
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract_topics(
            self,
            topic_model,
            documents: pd.DataFrame,
            c_tf_idf: csr_matrix,
            topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:

        updated_topics = {}
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, nr_repr_docs=6
        )

        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            # Prepare prompt
            truncated_docs = "\n".join(truncate_document(topic_model, 150, 'char', doc) for doc in docs)

            keywords = ', '.join([kw for kw, _ in topics[topic]][:10])

            prompt = (
                f"I have a topic with the following sentences:\n"
                f"{truncated_docs}\n\n"
                f"The topic is described by the following keywords: '{keywords}'.\n"
                "Based on this, respond with a short topic label:"
            )
            print(prompt)
            try:
                response = self.model.generate_content(prompt)
                print(response)
                label = response.text.strip()
                if not label:
                    label = "Unnamed Topic"
            except Exception as e:
                print(f"[Gemini error]: {e}")
                label = "Unknown"

            updated_topics[topic] = [(label, 1)]
        return updated_topics

def main():
    # Load or build embeddings
    embeddings, sentences = load_or_build_embeddings()

    print("Fitting new BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
        representation_model=GeminiRepresentation(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else KeyBERTInspired()
    )
    topic_assignment, probs = topic_model.fit_transform(sentences, embeddings)


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
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
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
    trim_size = 150
    short_texts = [s[:trim_size] + "..." if len(s) > trim_size else s for s in sentences]

    # Print preview
    label_previews = {}
    for label, snippet in zip(topic_labels, short_texts):
        label_previews.setdefault(label, [])
        if len(label_previews[label]) < 5:
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

    fig = px.scatter(df_plot, x="x", y="y",
                        color="topic",
                        title="3D Topic Visualization")
    fig.update_traces(marker=dict(size=3))
    fig.write_html("topic_viz_light.html", auto_open=False)
    fig.show()


if __name__ == "__main__":
    main()
