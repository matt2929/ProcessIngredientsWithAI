import os
import re
import requests
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Iterator, Mapping, Tuple, List, Set

from bertopic import BERTopic
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document
from platformdirs import user_data_dir
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
from umap import UMAP

# Assuming these exist in your local 'model.py'
from model import VideoObj, VideoType

# --- CONFIGURATION ---
APP_NAME = "MyMovieProject"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv"}
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# --- ML MODELS ---
embedding_model = SentenceTransformer("thenlper/gte-small")
umap_model = UMAP(n_neighbors=15, n_components=3, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    min_samples=2,
    metric="euclidean",
    cluster_selection_method="eom"
)


def sanitize_filename(name: str) -> str:
    """Removes characters that are illegal in file systems."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


class OllamaClient:
    """Dedicated client for efficient API communication."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session = requests.Session()

    def generate(self, prompt: str) -> str:
        payload = {"model": "llama3", "prompt": prompt, "stream": False}
        try:
            response = self.session.post(self.endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    def request_label(self, prompt: str) -> str:
        raw = self.generate(prompt)
        # Clean: first line only, remove quotes and numbering
        clean = raw.split('\n')[0]
        clean = re.sub(r'^(\d+\.\s+|Category:\s+)', '', clean).replace('"', '')
        return sanitize_filename(clean)


class OllamaRepresentation(BaseRepresentation):
    """BERTopic representation model using a local LLM."""

    def __init__(self, client: OllamaClient):
        self.client = client

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

        print("\n--- Starting Ollama Topic Labeling ---")
        for topic, docs in tqdm(repr_docs_mappings.items(), desc="Labeling topics"):
            truncated_docs = "\n".join(truncate_document(topic_model, 150, 'char', doc) for doc in docs)

            prompt = (
                f"I have these movie synopses:\n{truncated_docs}\n\n"
                "Identify ONE common genre or theme for this group.\n"
                "Constraint: Respond with ONLY the category name (2-4 words). No quotes, no lists."
            )

            label = self.client.request_label(prompt) or f"Topic_{topic}"
            print(f"[OLLAMA RESPONSE]: {label}")
            updated_topics[topic] = [(label, 1.0)]

        return updated_topics


class SerialShowIdentifier:
    """Filters out TV shows from movie processing."""

    def __init__(self):
        self.patterns = [re.compile(p, re.I) for p in [r"s[0-9]+e[0-9]+", r"^season", r"e[0-9]+"]]
        self.dir_cache: Set[Path] = set()

    def is_serial(self, root: Path, file: str) -> bool:
        if root in self.dir_cache or any(p.search(file) for p in self.patterns):
            self.dir_cache.add(root)
            return True
        return False


class MovieProgressSaver:
    """Handles persistence and state tracking using Pathlib."""

    def __init__(self):
        self.data_dir = Path(user_data_dir(APP_NAME))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "state.jsonl"
        print(f"Loading state file from {self.state_file}")
        self.seen_paths = self._load_seen_paths()

    def _load_seen_paths(self) -> set:
        if not self.state_file.exists():
            return set()

        paths = set()
        with self.state_file.open("r") as f:
            for line in f:
                try:
                    obj = VideoObj.model_validate_json(line)
                    paths.add(str(obj.path / obj.name_raw))
                except Exception:
                    continue
        return paths

    def read_videos(self) -> List[VideoObj]:
        if not self.state_file.exists():
            return []
        videos = []
        with self.state_file.open("r") as f:
            for line in f:
                videos.append(VideoObj.model_validate_json(line))
        return videos

    def save_all_videos(self, videos: List[VideoObj]):
        """Overwrites the state file with updated VideoObjects (e.g., with labels)."""
        with self.state_file.open("w") as f:
            for v in videos:
                f.write(v.model_dump_json() + "\n")

    def append_video(self, vid_obj: VideoObj):
        full_path = str(vid_obj.path / vid_obj.name_raw)
        if full_path not in self.seen_paths:
            with self.state_file.open("a") as f:
                f.write(vid_obj.model_dump_json() + "\n")
            self.seen_paths.add(full_path)


class MovieProcessor:
    """Orchestrates file scanning and ML analysis."""

    def __init__(self, ollama_client: OllamaClient, saver: MovieProgressSaver):
        self.client = ollama_client
        self.saver = saver
        self.ssi = SerialShowIdentifier()

    def scan_and_analyze(self, location: str):
        root_path = Path(location)
        files_to_process = []

        if not root_path.exists():
            print(f"Error: The path {location} does not exist.")
            return

        print("Scanning directory for new content (skipping inaccessible folders)...")

        # We use os.walk but convert to Path for the rest of your logic
        # this is much more stable for external drives than rglob
        for root, dirs, files in os.walk(root_path):
            try:
                curr_root = Path(root)
                for file in files:
                    p = curr_root / file
                    if p.suffix.lower() in VIDEO_EXTENSIONS:
                        if str(p) not in self.saver.seen_paths and not self.ssi.is_serial(p.parent, p.name):
                            files_to_process.append(p)
            except (PermissionError, FileNotFoundError) as e:
                print(f"Skipping inaccessible folder: {root}")
                continue

        if not files_to_process:
            print("Library update complete. No new movies found.")
        else:
            print(f"Found {len(files_to_process)} new movies to analyze.")
            for p in tqdm(files_to_process, desc="Updating Library", unit="movie"):
                video = VideoObj(name_raw=p.name, path=p.parent, video_type=VideoType.Movie)

                prompt = (
                    f"Based on the following filename associated to a movie '{p.name}', "
                    "give a brief synopsis including name, year, actors, genre, and themes"
                )
                video.synopsis = self.client.generate(prompt)
                self.saver.append_video(video)

        self.run_ml_pipeline()

    def run_ml_pipeline(self):
        videos = self.saver.read_videos()
        if not videos:
            print("No movie data found to cluster.")
            return

        synopses = [v.synopsis for v in videos]
        names = [v.name_raw for v in videos]

        print("Embedding descriptions and fitting BERTopic model...")
        embeddings = embedding_model.encode(synopses, show_progress_bar=True)

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True,
            representation_model=OllamaRepresentation(self.client)
        )

        topics, _ = topic_model.fit_transform(synopses, embeddings)
        reduced_embeddings = umap_model.fit_transform(embeddings)

        topic_info = topic_model.get_topic_info()
        topic_map = {
            row['Topic']: re.sub(r'^\d+_', '', row['Name'])
            for _, row in topic_info.iterrows()
        }

        # Pass 1: Set labels on VideoObjects and Prepare CSV data
        final_labels = []
        for i, video in enumerate(videos):
            label = topic_map.get(topics[i], "Uncategorized_Outliers")
            video.label = label  # SETTING THE LABEL ON THE VIDEO OBJECT
            final_labels.append(label)

        # Update the state file so labels persist
        self.saver.save_all_videos(videos)

        # CSV Export
        export_df = pd.DataFrame({
            "Movie_Title": names,
            "Original_Path": [str(v.path) for v in videos],
            "Suggested_Directory": final_labels
        })
        export_df.to_csv("movie_sorting_plan.csv", index=False)

        print("-" * 30)
        print("Sorting plan saved to movie_sorting_plan.csv")
        print("Labels have been updated in the state file.")
        print("-" * 30)

        # Plot
        df_plot = pd.DataFrame(reduced_embeddings, columns=["x", "y", "z"])
        df_plot["topic"] = final_labels
        df_plot["text"] = names

        fig = px.scatter_3d(
            df_plot, x="x", y="y", z="z",
            color="topic", hover_data=["text"],
            title="3D Movie Topic Visualization"
        )
        fig.update_traces(marker=dict(size=2))
        fig.show()


if __name__ == "__main__":
    scan_path = input("Enter directory path: ").strip()
    client = OllamaClient(OLLAMA_ENDPOINT)
    saver = MovieProgressSaver()

    processor = MovieProcessor(client, saver)
    processor.scan_and_analyze(scan_path)