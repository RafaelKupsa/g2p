import re
from nltk import word_tokenize
from g2p_model import G2PModel
from g2p_model_emb import G2PModelEmb


def get_dictionary():
    file = open("pronunciation_dictionary_clean.txt", "r", encoding="utf-8")
    pron_dict = {}
    for line in file.readlines():
        split_line = line[:-1].split(" ")
        if len(split_line) > 1:
            pron_dict[split_line[0]] = split_line[1:]
    file.close()
    return pron_dict


class G2PConverter:

    def __init__(self, embeddings=True, show_prediction_tag=False):
        self.model = G2PModelEmb() if embeddings else G2PModel()
        self.dictionary = get_dictionary()
        self.show_prediction_tag = show_prediction_tag

    def convert(self, text):
        words = self.pre_process(text)

        output = []
        for word in words:
            if word in self.dictionary:
                output.extend(self.dictionary[word])
            else:
                output.extend(self.model.predict(word))
                if self.show_prediction_tag:
                    output.append("(pred)")
            output.extend("|")

        return " ".join(output[:-1])

    def pre_process(self, text):
        text = re.sub("[,.?!]", " ", text)
        words = [re.sub("[^a-zäöüß]", "_", word.lower()) for word in text.split()]
        return words

