import argparse
import json
import os
import pickle

import numpy as np

from preprocessing_utils import load_classnames, load_labels
from utils import (DATA_FOLDER_PATH, FINETUNE_MODEL_PATH,
                   INTERMEDIATE_DATA_FOLDER_PATH, cosine_similarity_embeddings,
                   evaluate_predictions)


def evaluate(dataset, stage, suffix=None):
    data_dir = os.path.join(DATA_FOLDER_PATH, dataset)
    gold_labels = load_labels(data_dir)
    classes = load_classnames(data_dir)
    if stage == "Rep":
        with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset, f"document_repr_lm-{suffix}.pk"), "rb") as f:
            dictionary = pickle.load(f)
            document_representations = dictionary["document_representations"]
            class_representations = dictionary["class_representations"]
            repr_prediction = np.argmax(cosine_similarity_embeddings(document_representations, class_representations),
                                        axis=1)
            evaluate_predictions(gold_labels, repr_prediction)
    elif stage == "Align":
        with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset, f"data.{suffix}.pk"), "rb") as f:
            dictionary = pickle.load(f)
            documents_to_class = dictionary["documents_to_class"]
            evaluate_predictions(gold_labels, documents_to_class)
    else:
        with open(os.path.join(FINETUNE_MODEL_PATH, suffix, "eval_labels.json"), "r") as f:
            pred_labels = json.load(f)
            evaluate_predictions(gold_labels, pred_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--stage", type=str)
    parser.add_argument("--suffix", type=str)
    args = parser.parse_args()
    print(vars(args))
    evaluate(args.dataset, args.stage, args.suffix)
