# g2p - Grapheme to Phoneme Conversion for German

## Motivation

This project was realized for the seminar "Informationsverarbeitung I" (Conversational AI) at the LMU university of Munich.
The project's goal was to implement a python module capable of converting German text to a phonological representation.
This module could then theoretically be used as part of a speech synthesis pipeline in a larger conversational AI.

The G2P conversion works in the following way: The text is normalized and its individual words are looked up in a large pronunciation dictionary which was constructed specifically for this project from pronunciation data taken from Wiktionary. 
If the dictionary does not contain the word it is fed to a recurrent neural network that has been trained on said dictionary which will return a prediction for the word's pronunciation. The module allows the use of two different RNN models for comparison. 
Both use an encoder-decoder architecture with LSTM layers, one uses one-hot representations of words and pronunciations as input, the other uses extra layers to convert the words and pronunciations to embeddings.
The model using embedding layers performs much better.

For more information, read the article `Grapheme to Phoneme Conversion for German.pdf`

## Prerequisites

Python version 3.8

Python libraries:
- re (regular expressions)
- tensorflow (should come with keras and tensorboard)

If one wishes to recreate the dictionary and the model:
- requests
- bs4 (BeautifulSoup)
- os
- datetime
- random
- numpy
- sklearn


## Features

This project contains the following files and directories:
- `g2p_converter.py` (the main class): Contains the G2PConverter class which can be used to convert German words to their corresponding pronunciation
- `g2p_model.py`: Contains the G2PModel class which is called by G2PConverter if it was instantiated with embeddings set to false
- `g2p_model_emb.py`: Contains the G2PModelEmb class which is called by G2PConverter if it was instantiated with embeddings set to true
- `pronunciation_dictionary_clean.txt`: The pronunciation dictionary containing 35,307 word-pronunciation pairs that is used to look up pronunciations by G2PConverter
- `g2p_model_weights.hdf5`: The model weights for G2PModel
- `g2p_emb_model_weights.py`: The model weights for G2PModelEmb
- `g2p_model_train`: The training model architecture for G2PModel
- `g2p_model_test_enc`: The encoder testing model architecture for G2PModel
- `g2p_model_test_dec`: The decoder testing model architecture for G2PModel
- `g2p_model_emb_train`: The training model architecture for G2PModelEmb
- `g2p_model_emb_test_enc`: The encoder testing model architecture for G2PModelEmb
- `g2p_model_emb_test_dec`: The decoder testing model architecture for G2PModelEmb
- `create_dictionary.py`: The script that was used to collect the data for the dictionary
- `clean_up_dictionary.py`: The script that was used to clean up (normalize) the dictionary
- `pronunciation_dictionary.txt`: The pronunciation dictionary in its raw, non-normalized form
- `logs`: The folder containing training logs for both models which can be used to visualize the training process in tensorboard: While in the same folder containing the logs directory type `tensorboard --logdir logs` into the command line.
- `Grapheme to Phoneme Conversion for German.pdf`: The article that was written in accompaniment to the project for the seminar at LMU

## How to use?

To use the g2p conversion module you have to download at least the following files and directories but you can also just pull the whole g2p folder:
- `g2p_converter.py` (the main class)
- `g2p_model.py`
- `g2p_model_emb.py`
- `pronunciation_dictionary_clean.txt`
- `g2p_model_weights.hdf5`
- `g2p_emb_model_weights.py`
- `g2p_model_train`
- `g2p_model_test_enc`
- `g2p_model_test_dec`
- `g2p_model_emb_train`
- `g2p_model_emb_test_enc`
- `g2p_model_emb_test_dec`

The other files were used in the project's creation process and are there for documentation purposes for the seminar.

To directly use the g2p conversion tool you have to run a python 3.8 environment, import the G2PConverter class and instantiate an instance of it:

    >>> from g2p_converter import G2PConverter
    >>> c = G2PConverter()

Then you can use the `convert()` method to get the pronunciation of any text you want like this:

    >>> c.convert("Konvertiere diesen Text in seine Aussprache, bitte.")
    
    'K OH N V EH AX T II R EX | D II Z EX N | T EH K S T | IH N | Z AI N EX | AU S SH P R AA X EX | B IH T EX'

There are two options you can specify while making a G2PConverter instance. You can set `embeddings=False` which will load a model not trained on word embeddings. This is mainly for comparison with the other model only and does not provide very good predictions for unknown words.

    >>> c = G2PConverter(embeddings=False)
    
The second option is to set `show_prediction_tag=True` to get information about which pronunciations were predicted by the model and which ones were looked up in the pronunciation dictionary.

    >>> c = G2PConverter(show_prediction_tag=True)

## Notes
While making a new instance of the class G2PConverter, tensorflow might output several logging messages and/or several WARNINGS that the model is not compiled. These can be ignored, the model should work either way.
