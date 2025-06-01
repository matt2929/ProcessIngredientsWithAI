import os
import pickle
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ------------------- Config -------------------
CACHE_DIR = "./cache"
PRED_CACHE_PATH = os.path.join(CACHE_DIR, "pred_labels.pkl")
TRUE_CACHE_PATH = os.path.join(CACHE_DIR, "true_labels.pkl")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "test_embeddings.npy")

# Emotion labels
emotions_labels = [
    "happiness", "neutral", "sadness", "surprise", "love",
    "fear", "confusion", "disgust", "desire", "shame",
    "sarcasm", "anger", "guilt"
]


# ------------------- Utils -------------------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------- Evaluation -------------------
def evaluate_performance(y_true, y_pred, test_sentences):
    print("\n📊 Classification Report:\n")
    performance = classification_report(y_true, y_pred, labels=emotions_labels)
    print(performance)

    cm = confusion_matrix(y_true, y_pred, labels=emotions_labels)

    print("\n🔁 Most Frequent Confusions:\n")
    for i, label in enumerate(emotions_labels):
        row = cm[i].copy()
        row[i] = 0  # Ignore correct predictions
        if row.sum() == 0:
            print(f"{label}: No confusions.")
        else:
            most_confused_idx = row.argmax()
            confused_with = emotions_labels[most_confused_idx]
            print(f"{label} → {confused_with} ({row[most_confused_idx]} times)")

            print(f"Sample sentances where we expected {label} but got {confused_with}\n")
            for sample in miss_sampler(y_true=y_true,
                                       y_pred=y_pred,
                                       confused_with=confused_with,
                                       sample_from_label=label,
                                       test_sentences=test_sentences
                                       ):
                print("\t" + ("*" * 10))
                print(f"\t{sample}")
                print("\t" + ("*" * 10))
            print("===" * 10)


def miss_sampler(y_true, y_pred, confused_with, sample_from_label, test_sentences):
    sample_size = 2
    sampler = []
    for i, actual_label in enumerate(y_true):
        if y_pred[i] == confused_with and actual_label == sample_from_label:
            sampler.append(test_sentences[i])
            if len(sampler) >= sample_size:
                break
    return sampler


# ------------------- Main Pipeline -------------------
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load and shrink dataset
    print("📦 Loading dataset...")
    data = load_dataset("boltuix/emotions-dataset", split="train")
    data = data.shuffle(seed=42)  # smaller for test speed

    train_test_split = data.train_test_split(test_size=0.1, seed=42)
    test_sentences = train_test_split["test"]["Sentence"]
    true_labels = train_test_split["test"]["Label"]

    # Load or compute predictions
    if os.path.exists(PRED_CACHE_PATH) and os.path.exists(TRUE_CACHE_PATH):
        print("✅ Loading cached predictions...")
        pred_labels = load_pickle(PRED_CACHE_PATH)
        true_labels = load_pickle(TRUE_CACHE_PATH)
    else:
        print("🧠 Encoding and predicting...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        label_embeddings = model.encode(emotions_labels)
        test_embeddings = model.encode(test_sentences, show_progress_bar=True)

        sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
        y_pred = np.argmax(sim_matrix, axis=1)
        pred_labels = [emotions_labels[i] for i in y_pred]

        save_pickle(pred_labels, PRED_CACHE_PATH)
        save_pickle(true_labels, TRUE_CACHE_PATH)

    evaluate_performance(true_labels, pred_labels, test_sentences)


if __name__ == "__main__":
    main()
