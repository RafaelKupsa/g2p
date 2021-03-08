import os
import random
import numpy as np
from tensorflow.keras.models import load_model


def get_dictionary():
    file = open("pronunciation_dictionary_clean.txt", "r", encoding="utf-8")
    pron_dict = []
    for line in file.readlines():
        split_line = line[:-1].split(" ")
        if 2 <= len(split_line) <= 25:
            pron_dict.append((split_line[0], split_line[1:]))
    file.close()

    random.seed(10)
    random.shuffle(pron_dict)

    return pron_dict


class G2PModel:

    def __init__(self):

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # data
        self.UNKNOWN_G = "_"
        self.PADDING_P = ""
        self.START_P = "^"
        self.END_P = "$"

        self.GRAPHEMES = list("abcdefghijklmnopqrstuvwxyzäöüß") + [self.UNKNOWN_G]
        self.PHONEMES = ["A", "AA", "E", "EE", "EH", "AEE", "I", "II", "IH",
                         "O", "OO", "OH", "OE", "OEE", "OEH", "U", "UU", "UH", "UE", "UEE", "UEH",
                         "AI", "AU", "OI", "AX", "EX", "B", "C", "D", "JH", "F", "G", "H",
                         "J", "K", "L", "M", "N", "NG", "P", "PF", "R", "S", "SH", "T", "TS",
                         "CH", "V", "X", "Z", "ZH", "?", self.PADDING_P, self.START_P, self.END_P]

        self.G2IDX = {g: idx for idx, g in enumerate(self.GRAPHEMES)}
        self.IDX2G = {idx: g for idx, g in enumerate(self.GRAPHEMES)}

        self.P2IDX = {p: idx for idx, p in enumerate(self.PHONEMES)}
        self.IDX2P = {idx: p for idx, p in enumerate(self.PHONEMES)}

        self.WORD2PRON = get_dictionary()

        self.WORDS = [list(element[0]) for element in self.WORD2PRON]
        self.PRONS = [[self.START_P] + element[1] + [self.END_P] for element in self.WORD2PRON]

        self.MAX_GRAPHEMES = max([len(element) for element in self.WORDS])
        self.MAX_PHONEMES = max([len(element) for element in self.PRONS])

        # model parameters
        self.BATCH_SIZE = 64
        self.HIDDEN_NODES = 256
        self.EPOCHS = 150
        self.LEARNING_RATE = 0.0003
        self.MODEL_WEIGHTS_FILEPATH = "g2p_model_weights.hdf5"

        # model
        self.model_train = load_model("g2p_model_train")
        self.model_test_enc = load_model("g2p_model_test_enc")
        self.model_test_dec = load_model("g2p_model_test_dec")

        self.model_train.load_weights(self.MODEL_WEIGHTS_FILEPATH)

    def predict(self, word):

        # convert word to one-hot matrix
        word_onehot = np.zeros((self.MAX_GRAPHEMES, len(self.GRAPHEMES)))
        for i, g in enumerate(word):
            word_onehot[i, :] = self.g2onehot(g)
        word_onehot = np.array([word_onehot])

        # feed the model
        state_vectors = self.model_test_enc.predict(word_onehot)
        prev_phoneme = np.zeros((1, 1, len(self.PHONEMES)))
        prev_phoneme[0, 0, self.P2IDX[self.START_P]] = 1.
        pron_complete = False
        pron_prediction = []

        while not pron_complete:
            phoneme_onehot, state_h, state_c = self.model_test_dec.predict([prev_phoneme] + state_vectors)
            phoneme_idx = np.argmax(phoneme_onehot[0, -1, :])
            phoneme = self.IDX2P[phoneme_idx]
            pron_prediction.append(phoneme)

            if phoneme == self.END_P or len(pron_prediction) >= self.MAX_PHONEMES:
                pron_complete = True

            prev_phoneme = np.zeros((1, 1, len(self.PHONEMES)))
            prev_phoneme[0, 0, phoneme_idx] = 1.
            state_vectors = [state_h, state_c]

        return pron_prediction[:-1]

    def g2onehot(self, grapheme):
        onehot = np.zeros((len(self.GRAPHEMES)))
        onehot[self.G2IDX[grapheme]] = 1.
        return onehot
