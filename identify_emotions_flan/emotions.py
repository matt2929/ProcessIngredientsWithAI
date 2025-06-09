from time import time

from datasets import load_dataset, tqdm, ClassLabel
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline
from transformers.pipelines.base import KeyDataset
import torch
import difflib

# Emotion labels
emotions_labels = [
    "happiness", "neutral", "sadness", "surprise", "love", "fear", "confusion",
    "disgust", "desire", "shame", "sarcasm", "anger", "guilt"
]

def match_label(text, label_set):
    match = difflib.get_close_matches(text, label_set, n=1, cutoff=0.8)
    if text=="joy":
        return "happiness"
    return match[0] if match else None

def main():
    # Load dataset
    data = load_dataset("boltuix/emotions-dataset", split="train")
    test_data = data.cast_column("Label", ClassLabel(names=emotions_labels))

    data = test_data.train_test_split(test_size=0.001, seed=42, stratify_by_column="Label")
    _, test_data = data["train"], data["test"]

    # Train classifier
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,
        max_length=10,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    prompt = f"Pick only one of the following emotions: {', '.join(emotions_labels)}. Which best describes the following sentence? "
    prompted_test_data = test_data.map(lambda example: {"t5": prompt + example["Sentence"]})
    y_pred = []
    total_failure = 0
    for batch in tqdm(pipe(KeyDataset(prompted_test_data, "t5"), batch_size=2),
                      total=len(prompted_test_data)):

        torch.mps.empty_cache()
        for output in batch:
            text = output["generated_text"].strip().lower()
            pred_label = match_label(text, emotions_labels)
            if not pred_label:
                total_failure += 1
                y_pred.append(text)  # Still append raw for analysis
                print(f"{text}")
            else:
                y_pred.append(pred_label)
    print(f"Couldn't figure out: {total_failure}/{len(prompted_test_data)}")
    y_true_str = [emotions_labels[label] if isinstance(label, int) else label for label in test_data['Label']]
    evaluate_performance(y_true_str, y_pred, test_data["Sentence"])


# Evaluate performance
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


main()
