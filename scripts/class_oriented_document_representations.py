import argparse
import os
import pickle as pk
from collections import defaultdict

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy
from tqdm import tqdm

from static_representations import handle_sentence
from utils import (INTERMEDIATE_DATA_FOLDER_PATH, MODELS,
                   cosine_similarity_embedding, cosine_similarity_embeddings,
                   evaluate_predictions, tensor_to_numpy)


def probability_confidence(prob):
    return max(softmax(prob))


def rank_by_significance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(softmax(similarity)) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking


def rank_by_relation(embeddings, class_embeddings):
    relation_score = cosine_similarity_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1))
    relation_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(relation_score)))}
    return relation_ranking


def mul(l):
    m = 1
    for x in l:
        m *= x + 1
    return m


def average_with_harmonic_series(representations):
    weights = [0.0] * len(representations)
    for i in range(len(representations)):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=0)


def weights_from_ranking(rankings):
    if len(rankings) == 0:
        assert False
    if type(rankings[0]) == type(0):
        rankings = [rankings]
    rankings_num = len(rankings)
    rankings_len = len(rankings[0])
    assert all(len(rankings[i]) == rankings_len for i in range(rankings_num))
    total_score = []
    for i in range(rankings_len):
        total_score.append(mul(ranking[i] for ranking in rankings))

    total_ranking = {i: r for r, i in enumerate(np.argsort(np.array(total_score)))}
    if rankings_num == 1:
        assert all(total_ranking[i] == rankings[0][i] for i in total_ranking.keys())
    weights = [0.0] * rankings_len
    for i in range(rankings_len):
        weights[i] = 1. / (total_ranking[i] + 1)
    return weights


def weight_sentence_with_attention(vocab, tokenized_text, contextualized_word_representations, class_representations,
                                   attention_mechanism):
    assert len(tokenized_text) == len(contextualized_word_representations)

    contextualized_representations = []
    static_representations = []

    static_word_representations = vocab["static_word_representations"]
    word_to_index = vocab["word_to_index"]
    for i, token in enumerate(tokenized_text):
        if token in word_to_index:
            static_representations.append(static_word_representations[word_to_index[token]])
            contextualized_representations.append(contextualized_word_representations[i])
    if len(contextualized_representations) == 0:
        print("Empty Sentence (or sentence with no words that have enough frequency)")
        return np.average(contextualized_word_representations, axis=0)

    significance_ranking = rank_by_significance(contextualized_representations, class_representations)
    relation_ranking = rank_by_relation(contextualized_representations, class_representations)
    significance_ranking_static = rank_by_significance(static_representations, class_representations)
    relation_ranking_static = rank_by_relation(static_representations, class_representations)
    if attention_mechanism == "none":
        weights = [1.0] * len(contextualized_representations)
    elif attention_mechanism == "significance":
        weights = weights_from_ranking(significance_ranking)
    elif attention_mechanism == "relation":
        weights = weights_from_ranking(relation_ranking)
    elif attention_mechanism == "significance_static":
        weights = weights_from_ranking(relation_ranking)
    elif attention_mechanism == "relation_static":
        weights = weights_from_ranking(relation_ranking)
    elif attention_mechanism == "mixture":
        weights = weights_from_ranking((significance_ranking,
                                        relation_ranking,
                                        significance_ranking_static,
                                        relation_ranking_static))
    else:
        assert False
    return np.average(contextualized_representations, weights=weights, axis=0)


def weight_sentence(model,
                    vocab,
                    tokenization_info,
                    class_representations,
                    attention_mechanism,
                    layer
                    ):
    tokenized_text, tokenized_to_id_indicies, tokenids_chunks = tokenization_info
    contextualized_word_representations = handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies,
                                                          tokenids_chunks)
    document_representation = weight_sentence_with_attention(vocab, tokenized_text, contextualized_word_representations,
                                                             class_representations, attention_mechanism)
    return document_representation


def main(args):
    data_folder = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, args.dataset_name)
    with open(os.path.join(data_folder, "dataset.pk"), "rb") as f:
        dataset = pk.load(f)
        class_names = dataset["class_names"]

    static_repr_path = os.path.join(data_folder, f"static_repr_lm-{args.lm_type}-{args.layer}.pk")
    with open(static_repr_path, "rb") as f:
        vocab = pk.load(f)
        static_word_representations = vocab["static_word_representations"]
        word_to_index = vocab["word_to_index"]
        vocab_words = vocab["vocab_words"]

    with open(os.path.join(data_folder, f"tokenization_lm-{args.lm_type}-{args.layer}.pk"), "rb") as f:
        tokenization_info = pk.load(f)["tokenization_info"]

    print("Finish reading data")

    print(class_names)

    finished_class = set()
    masked_words = set(class_names)
    cls_repr = [None for _ in range(len(class_names))]
    class_words = [[class_names[cls]] for cls in range(len(class_names))]
    class_words_representations = [[static_word_representations[word_to_index[class_names[cls]]]]
                                   for cls in range(len(class_names))]
    for t in range(1, args.T):
        class_representations = [average_with_harmonic_series(class_words_representation)
                                 for class_words_representation in class_words_representations]
        cosine_similarities = cosine_similarity_embeddings(static_word_representations,
                                                           class_representations)
        nearest_class = cosine_similarities.argmax(axis=1)
        similarities = cosine_similarities.max(axis=1)
        for cls in range(len(class_names)):
            if cls in finished_class:
                continue
            highest_similarity = -1.0
            highest_similarity_word_index = -1
            lowest_masked_words_similarity = 1.0
            existing_class_words = set(class_words[cls])
            stop_criterion = False
            for i, word in enumerate(vocab_words):
                if nearest_class[i] == cls:
                    if word not in masked_words:
                        if similarities[i] > highest_similarity:
                            highest_similarity = similarities[i]
                            highest_similarity_word_index = i
                    else:
                        if word not in existing_class_words:
                            stop_criterion = True
                            break
                        lowest_masked_words_similarity = min(lowest_masked_words_similarity, similarities[i])
                else:
                    if word in existing_class_words:
                        stop_criterion = True
                        break
            # the topmost t words are no longer the t words in class_words
            if lowest_masked_words_similarity < highest_similarity:
                stop_criterion = True

            if stop_criterion:
                finished_class.add(cls)
                class_words[cls] = class_words[cls][:-1]
                class_words_representations[cls] = class_words_representations[cls][:-1]
                cls_repr[cls] = average_with_harmonic_series(class_words_representations[cls])
                print(class_words[cls])
                break
            class_words[cls].append(vocab_words[highest_similarity_word_index])
            class_words_representations[cls].append(static_word_representations[highest_similarity_word_index])
            masked_words.add(vocab_words[highest_similarity_word_index])
            cls_repr[cls] = average_with_harmonic_series(class_words_representations[cls])
        if len(finished_class) == len(class_names):
            break
            
    class_representations = np.array(cls_repr)
    model_class, tokenizer_class, pretrained_weights = MODELS[args.lm_type]
    model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
    model.eval()
    model.cuda()
    document_representations = []
    for i, _tokenization_info in tqdm(enumerate(tokenization_info), total=len(tokenization_info)):
        document_representation = weight_sentence(model,
                                                  vocab,
                                                  _tokenization_info,
                                                  class_representations,
                                                  args.attention_mechanism,
                                                  args.layer)
        document_representations.append(document_representation)
    document_representations = np.array(document_representations)
    print("Finish getting document representations")
    with open(os.path.join(data_folder,
                           f"document_repr_lm-{args.lm_type}-{args.layer}-{args.attention_mechanism}-{args.T}.pk"),
              "wb") as f:
        pk.dump({
            "class_words": class_words,
            "class_representations": class_representations,
            "document_representations": document_representations,
        }, f, protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--attention_mechanism", type=str, default="mixture")

    args = parser.parse_args()
    print(vars(args))
    main(args)
