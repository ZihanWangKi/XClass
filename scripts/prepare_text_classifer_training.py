import argparse
import itertools
import json
import operator
import os
import pickle
from shutil import copyfile

import numpy as np

from preprocessing_utils import load_clean_text
from utils import (DATA_FOLDER_PATH, INTERMEDIATE_DATA_FOLDER_PATH,
                   evaluate_predictions, most_common)


def write_to_dir(text, labels, dataset_name, suffix_name):
    assert len(text) == len(labels)
    new_dataset_name = f"{dataset_name}_{suffix_name}"
    # removes all potentially cached files in it
    if os.path.isdir(os.path.join(DATA_FOLDER_PATH, new_dataset_name)):
        assert False, f"{os.path.join(DATA_FOLDER_PATH, new_dataset_name)} exists."
    os.makedirs(os.path.join(DATA_FOLDER_PATH, new_dataset_name), exist_ok=True)
    with open(os.path.join(DATA_FOLDER_PATH, new_dataset_name, "dataset.txt"), "w") as f:
        for i, line in enumerate(text):
            f.write(line)
            f.write("\n")

    with open(os.path.join(DATA_FOLDER_PATH, new_dataset_name, "labels.txt"), "w") as f:
        for i, line in enumerate(labels):
            f.write(str(line))
            f.write("\n")
    copyfile(os.path.join(DATA_FOLDER_PATH, dataset_name, "classes.txt"),
             os.path.join(DATA_FOLDER_PATH, new_dataset_name, "classes.txt"))


def main(dataset_name, suffix, confidence_threshold):
    data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)

    cleaned_text = load_clean_text(os.path.join(DATA_FOLDER_PATH, dataset_name))

    with open(os.path.join(data_dir, f"data.{suffix}.pk"), "rb") as f:
        save_data = pickle.load(f)
        documents_to_class = save_data["documents_to_class"]
        distance = save_data["distance"]
        num_classes = distance.shape[1]
    pseudo_document_class_with_confidence = [[] for _ in range(num_classes)]
    for i in range(documents_to_class.shape[0]):
        pseudo_document_class_with_confidence[documents_to_class[i]].append((distance[i][documents_to_class[i]], i))

    selected = []
    ###
    gold_labels = list(map(int, open(os.path.join(DATA_FOLDER_PATH, dataset_name, "labels.txt")).readlines()))
    ###
    for i in range(num_classes):
        pseudo_document_class_with_confidence[i] = sorted(pseudo_document_class_with_confidence[i])
        num_docs_to_take = int(len(pseudo_document_class_with_confidence[i]) * confidence_threshold)
        confident_documents = pseudo_document_class_with_confidence[i][:num_docs_to_take]
        confident_documents = [x[1] for x in confident_documents]
        selected.extend(confident_documents)

    selected = sorted(selected)
    text = [cleaned_text[i] for i in selected]
    classes = [documents_to_class[i] for i in selected]
    ###
    gold_classes = [gold_labels[i] for i in selected]
    evaluate_predictions(gold_classes, classes)
    ###
    write_to_dir(text, classes, dataset_name, f"{suffix}.{confidence_threshold}")
    # json.dump(selected, open("m_sel.json", "w"))
    # json.dump(documents_to_class.tolist(), open("m_pre.json", "w"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="agnews")
    parser.add_argument("--suffix", type=str, default="pca64.clusgmm.bbu-12.mixture-100.42")
    parser.add_argument("--confidence_threshold", default=0.5)
    args = parser.parse_args()
    print(vars(args))
    main(args.dataset_name, args.suffix, args.confidence_threshold)
