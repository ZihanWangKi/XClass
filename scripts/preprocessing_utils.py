import csv
import itertools
import os
import random
import re
from collections import Counter

import numpy as np
from tqdm import tqdm

from utils import DATA_FOLDER_PATH


# mainly for agnews
def clean_html(string: str):
    left_mark = '&lt;'
    right_mark = '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = string.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = string.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + string)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        clean_html.clean_links.append(string[next_left_start: next_right_start + len(right_mark)])
        string = string[:next_left_start] + " " + string[next_right_start + len(right_mark):]
    return string


clean_html.clean_links = []


# mainly for 20news
def clean_email(string: str):
    return " ".join([s for s in string.split() if "@" not in s])


def clean_str(string):
    string = clean_html(string)
    string = clean_email(string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def load_clean_text(data_dir):
    text = load_text(data_dir)
    return [clean_str(doc) for doc in text]


def load_text(data_dir):
    with open(os.path.join(data_dir, 'dataset.txt'), mode='r', encoding='utf-8') as text_file:
        text = list(map(lambda x: x.strip(), text_file.readlines()))
    return text


def load_labels(data_dir):
    with open(os.path.join(data_dir, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: int(x.strip()), label_file.readlines()))
    return labels


def load_classnames(data_dir):
    with open(os.path.join(data_dir, 'classes.txt'), mode='r', encoding='utf-8') as classnames_file:
        class_names = "".join(classnames_file.readlines()).strip().split("\n")
    return class_names


def text_statistics(text, name="default"):
    sz = len(text)

    tmp_text = [s.split(" ") for s in text]
    tmp_list = [len(doc) for doc in tmp_text]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print(f"\n### Dataset statistics for {name}: ###")
    print('# of documents is: {}'.format(sz))
    print('Document max length: {} (words)'.format(len_max))
    print('Document average length: {} (words)'.format(len_avg))
    print('Document length std: {} (words)'.format(len_std))
    print(f"#######################################")


def load(dataset_name):
    data_dir = os.path.join(DATA_FOLDER_PATH, dataset_name)
    text = load_text(data_dir)
    class_names = load_classnames(data_dir)
    text = [s.strip() for s in text]
    text_statistics(text, "raw_txt")

    cleaned_text = [clean_str(doc) for doc in text]
    print(f"Cleaned {len(clean_html.clean_links)} html links")
    text_statistics(cleaned_text, "cleaned_txt")

    result = {
        "class_names": class_names,
        "raw_text": text,
        "cleaned_text": cleaned_text,
    }
    return result


if __name__ == '__main__':
    data = load('agnews')
