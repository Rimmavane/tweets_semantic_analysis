from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from hyperparameters import *

def create_encoder(list_of_texts):
    """
    Creates encoder that creates a vocabulary based on given list of texts.
    It can be used as a parameter for create_model() function.
    """
    encoder = TextVectorization(max_tokens=NUM_WORDS)
    encoder.adapt(list_of_texts)
    return encoder

def create_model(encoder):
    """
    Creates and returns a model
    """
    # Define the model architecture
    model = Sequential()
    model.add(encoder)
    model.add(Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=128,
        # Use masking to handle the variable sequence lengths
        mask_zero=True))
    model.add(LSTM(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model

