import os
import random
import numpy as np
import re
import datetime

from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split


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


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

GRAPHEMES = list("abcdefghijklmnopqrstuvwxyzäöüß") + ["_"]  # _ is unknown

PHONEMES = ["A", "AA", "E", "EE", "EH", "AEE", "I", "II", "IH",
            "O", "OO", "OH", "OE", "OEE", "OEH", "U", "UU", "UH", "UE", "UEE", "UEH",
            "AI", "AU", "OI", "AX", "EX", "B", "C", "D", "JH", "F", "G", "H",
            "J", "K", "L", "M", "N", "NG", "P", "PF", "R", "S", "SH", "T", "TS",
            "CH", "V", "X", "Z", "ZH", "?", "", "^", "$"]  # ^ is start symbol, $ is end symbol

G2IDX = {g: idx for idx, g in enumerate(GRAPHEMES)}
IDX2G = {idx: g for idx, g in enumerate(GRAPHEMES)}

P2IDX = {p: idx for idx, p in enumerate(PHONEMES)}
IDX2P = {idx: p for idx, p in enumerate(PHONEMES)}

WORD2PRON = get_dictionary()

INPUT_WORDS = [list(element[0]) for element in WORD2PRON]
OUTPUT_PRONS = [["^"] + element[1] + ["$"] for element in WORD2PRON]

INPUT_MAX_CHARS = max([len(element) for element in INPUT_WORDS])
OUTPUT_MAX_CHARS = max([len(element) for element in OUTPUT_PRONS])

# model parameters
BATCH_SIZE = 64
HIDDEN_NODES = 256
EPOCHS = 150
LEARNING_RATE = 0.0003
MODEL_WEIGHTS_FILEPATH = "g2p_model_weights.hdf5"

SHOULD_SAVE = False
SHOULD_TRAIN = False


def g2onehot(grapheme):
    vec = np.zeros((len(GRAPHEMES)))
    vec[G2IDX[grapheme]] = 1.
    return vec


def p2onehot(phoneme):
    vec = np.zeros((len(PHONEMES)))
    vec[P2IDX[phoneme]] = 1.
    return vec


def onehot2g(grapheme_onehot):
    if np.count_nonzero(grapheme_onehot) == 0:
        return ""
    return IDX2G[np.argmax(grapheme_onehot)]


def onehot2p(phoneme_onehot):
    if np.count_nonzero(phoneme_onehot) == 0:
        return ""
    return IDX2P[np.argmax(phoneme_onehot)]


def make_model_data():

    input_matrix = []
    output_matrix = []

    for word, pron in zip(INPUT_WORDS, OUTPUT_PRONS):

        word_matrix = np.zeros((INPUT_MAX_CHARS, len(GRAPHEMES)))
        for i, g in enumerate(word):
            word_matrix[i, :] = g2onehot(g)
        pron_matrix = np.zeros((OUTPUT_MAX_CHARS, len(PHONEMES)))
        for i, p in enumerate(pron):
            pron_matrix[i, :] = p2onehot(p)

        input_matrix.append(word_matrix)
        output_matrix.append(pron_matrix)

    input_matrix = np.array(input_matrix)
    output_matrix = np.array(output_matrix)
    output_matrix_shifted = np.pad(output_matrix, ((0, 0), (0, 1), (0, 0)), mode="constant")[:, 1:, :]

    return input_matrix, output_matrix, output_matrix_shifted


# make training and test models
def make_models():
    encoder_inputs = Input(shape=(None, len(GRAPHEMES)))
    encoder = LSTM(HIDDEN_NODES, return_state=True)

    decoder_inputs = Input(shape=(None, len(PHONEMES)))
    decoder = LSTM(HIDDEN_NODES, return_sequences=True, return_state=True)
    decoder_dense = Dense(len(PHONEMES), activation="softmax")

    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]  # hidden state, cell state
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_prediction = decoder_dense(decoder_outputs)

    training_mdl = Model([encoder_inputs, decoder_inputs], decoder_prediction)

    testing_encoder_mdl = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(HIDDEN_NODES,))
    decoder_state_input_c = Input(shape=(HIDDEN_NODES,))
    decoder_state_inputs = [decoder_state_input_c, decoder_state_input_h]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    testing_decoder_prediction = decoder_dense(decoder_outputs)

    testing_decoder_mdl = Model([decoder_inputs] + decoder_state_inputs, [testing_decoder_prediction] + decoder_states)

    return training_mdl, testing_encoder_mdl, testing_decoder_mdl


def train(model, weights_path, enc_input, dec_input, dec_output):
    checkpoint = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor="val_loss", patience=5)
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    optimizer = Adam(lr=LEARNING_RATE)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics="accuracy")
    model.fit([enc_input, dec_input],
              dec_output,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2,
              callbacks=[checkpoint, stopper, tensorboard_callback])


def predict(input_onehot, enc, dec):
    state_vectors = enc.predict(input_onehot)
    prev_phoneme = np.zeros((1, 1, len(PHONEMES)))
    prev_phoneme[0, 0, P2IDX["^"]] = 1.

    end_found = False
    pron = []
    while not end_found:
        dec_output, h, c = dec.predict([prev_phoneme] + state_vectors)

        predicted_phoneme_idx = np.argmax(dec_output[0, -1, :])
        predicted_phoneme = IDX2P[predicted_phoneme_idx]

        pron.append(predicted_phoneme)

        if predicted_phoneme == "$" or len(pron) > OUTPUT_MAX_CHARS:
            end_found = True

        prev_phoneme = np.zeros((1, 1, len(PHONEMES)))
        prev_phoneme[0, 0, predicted_phoneme_idx] = 1.
        state_vectors = [h, c]

    return pron


def test_sample(count):
    correct_count = 0

    for i in random.sample(range(len(encoder_input_test)), count):

        example_onehot = encoder_input_test[i:i + 1]
        example_word = [onehot2g(onehot) for onehot in example_onehot[0]]
        example_word = "".join(example_word)

        correct_pron_onehot = decoder_input_test[i:i + 1]
        correct_pron = [onehot2p(onehot) for onehot in correct_pron_onehot[0]]
        correct_pron = " ".join(correct_pron).strip()
        correct_pron = correct_pron[2:-2]

        predicted_pron = predict(example_onehot, testing_encoder_model, testing_decoder_model)
        predicted_pron = " ".join(predicted_pron[:-1])

        is_correct = correct_pron == predicted_pron
        if is_correct:
            correct_count += 1

        print(example_word, "-->", correct_pron, "-- prediction:", predicted_pron, "-- correct?", is_correct)

    print("Correct:", str(correct_count) + "/" + str(count) + ",", str(correct_count / count) + "%")


def test_custom(word):
    example_onehot = np.zeros((INPUT_MAX_CHARS, len(GRAPHEMES)))
    for i, g in enumerate(list(re.sub("[^a-zäöüß]", "_", word.lower()))):
        example_onehot[i, :] = g2onehot(g)

    example_onehot = np.array([example_onehot])
    predicted_pron = predict(example_onehot, testing_encoder_model, testing_decoder_model)
    print(word, "-->", " ".join(predicted_pron[:-1]))


encoder_input, decoder_input, decoder_output = make_model_data()

training_model, testing_encoder_model, testing_decoder_model = make_models()

(encoder_input_train, encoder_input_test,
 decoder_input_train, decoder_input_test,
 decoder_output_train, decoder_output_test) = train_test_split(
    encoder_input, decoder_input, decoder_output,
    test_size=0.2, random_state=10)

# TRAINING
if SHOULD_TRAIN:
    train(training_model, MODEL_WEIGHTS_FILEPATH, encoder_input_train, decoder_input_train, decoder_output_train)

if SHOULD_SAVE:
    training_model.load_weights(MODEL_WEIGHTS_FILEPATH)
    training_model.save("g2p_model_train")
    testing_encoder_model.save("g2p_model_test_enc")
    testing_decoder_model.save("g2p_model_test_dec")

# TESTING
training_model.load_weights(MODEL_WEIGHTS_FILEPATH)
test_sample(100)