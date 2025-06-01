from transformers import pipeline
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from itertools import islice

def main():
    # Load model and tokenizer
    data = load_dataset("rotten_tomatoes")
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    pipe = pipeline(
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device="mps"
    )

    y_pred = []
    for i, output in enumerate(pipe(KeyDataset(data["test"], "text"))):
        negative_score = output[0]["score"]
        pos_score = output[2]["score"]
        assignment = np.argmax([negative_score, pos_score])
        y_pred.append(assignment)
        if data["test"][i]['label'] != assignment:
            print(f"""
Failure to identify:
{data["test"][i]['text']}
pos: {pos_score}
neg: {negative_score}
""")
    evaluate_performance(data["test"]["label"], y_pred)


def evaluate_performance(y_true, y_pred):
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)


main()
