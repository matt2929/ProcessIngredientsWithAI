import os
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Emotion labels
emotions_data_set = [
    "happiness", "neutral", "sadness", "surprise", "love", "fear", "confusion",
    "disgust", "desire", "shame", "sarcasm", "anger", "guilt"
]

# Paths for caching
CACHE_DIR = "./embedding_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
X_train_path = os.path.join(CACHE_DIR, "X_train.npy")
X_test_path = os.path.join(CACHE_DIR, "X_test.npy")
y_train_path = os.path.join(CACHE_DIR, "y_train.npy")
y_test_path = os.path.join(CACHE_DIR, "y_test.npy")
pred_labels_path = os.path.join(CACHE_DIR, "pred_labels.npy")

# Load dataset
data = load_dataset("boltuix/emotions-dataset", split="train")
data = data.train_test_split(test_size=0.1, seed=42)
train_data, test_data = data["train"], data["test"]

# Extract sentences and labels
train_sentences = train_data["Sentence"]
test_sentences = test_data["Sentence"]
label_to_index = {label: idx for idx, label in enumerate(emotions_data_set)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
train_labels = [label_to_index[label] for label in train_data["Label"]]
test_labels = [label_to_index[label] for label in test_data["Label"]]

# Load SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Encode and cache embeddings
if os.path.exists(X_train_path):
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
else:
    X_train = model.encode(train_sentences, show_progress_bar=True)
    X_test = model.encode(test_sentences, show_progress_bar=True)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

# Train classifier
clf = LogisticRegression(max_iter=1000, verbose=1)
clf.fit(X_train, y_train)

# Predict and cache
if os.path.exists(pred_labels_path):
    y_pred = np.load(pred_labels_path)
else:
    y_pred = clf.predict(X_test)
    np.save(pred_labels_path, y_pred)


# Evaluate performance
def evaluate_performance(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=emotions_data_set
    ))

    print("\nMost confused pairs:")
    confusion = {}
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            key = (true, pred)
            confusion[key] = confusion.get(key, 0) + 1
    most_confused = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
    for (true, pred), count in most_confused[:5]:
        print(f"{emotions_data_set[true]} → {emotions_data_set[pred]}: {count} times")

    # Print two example sentences from most confused class
    if most_confused:
        most_confused_true, most_confused_pred = most_confused[0][0]
        example_idxs = [i for i, (t, p) in enumerate(zip(y_test, y_pred))
                        if t == most_confused_true and p == most_confused_pred][:2]
        print(f"\nExample sentences where {emotions_data_set[most_confused_true]} "
              f"was misclassified as {emotions_data_set[most_confused_pred]}:")
        for i in example_idxs:
            print(f" - {test_sentences[i]}")


evaluate_performance(y_test, y_pred)


# Visualize classification
def visualize_misclassified(X_test, true_labels, pred_labels, title="Misclassified Highlighted"):
    print("🧪 Reducing dimensions for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(X_test)

    # Convert to numpy arrays of label strings
    true_str = np.array([emotions_data_set[i] for i in true_labels])
    pred_str = np.array([emotions_data_set[i] for i in pred_labels])
    is_misclassified = true_labels != pred_labels

    plt.figure(figsize=(14, 10))

    # All points
    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=true_str,
        palette="tab20",
        alpha=0.4,
        s=30,
        legend='brief'
    )

    # Misclassified overlay
    plt.scatter(
        reduced[is_misclassified, 0],
        reduced[is_misclassified, 1],
        facecolors='none',
        edgecolors='red',
        linewidths=1.5,
        s=80,
        label='Misclassified'
    )

    plt.title(title)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


visualize_misclassified(X_test, y_test, y_pred)
