import transformers
import torch
from utils import MODELS

# an alternative way to obtain representation (through a template)
class InitialClassRepresentationObtainer:
    def __init__(self, lm_type):
        model_class, tokenizer_class, pretrained_weights = MODELS[lm_type]
        self.model = model_class.from_pretrained(pretrained_weights)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    def get_representations_for_sentence(self, sentence):
        sentence = sentence.strip()
        char_offset_to_word_index = {}
        words = []
        n_chars = 0
        for i, word in enumerate(sentence.split()):
            char_offset_to_word_index[n_chars] = i
            words.append(word)
            n_chars += len(word) + 1
        word_representations = [None for _ in range(len(words))]

        token_ids = self.tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True)
        offset_mapping = token_ids["offset_mapping"].numpy()[0]
        del token_ids["offset_mapping"]
        with torch.no_grad():
            output = self.model(**token_ids,
                                output_hidden_states=True,
                                return_dict=True)
            representations = output.hidden_states[12].squeeze(0).numpy()
            for i, (start, end) in enumerate(offset_mapping):
                if start == end == 0:
                    continue
                if start in char_offset_to_word_index:
                    word_representations[char_offset_to_word_index[start]] = representations[i]
            assert all(x is not None for x in word_representations)
        return words, word_representations

    def get_representation_for_classname(self, classname):
        words, word_representations = get_representations_for_sentence(
            "This document is about {} .".format(classname),
            model,
            tokenizer)
        return np.average(word_representations[4: -1], axis=0)
