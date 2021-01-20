from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import KFold

import numpy as np

from model import create_model, create_encoder
from data_loading import load_data, clean_sentence
from plot_results import HistoriesStorage, plot_model_histories
from hyperparameters import *

tweets_data = load_data('Data_tweets.csv')
tweets_data = tweets_data[[1, 6]]
tweets_data.columns = ['Class', 'Tweet']

# One-hot-encoding classes
def substitute_classes(x):
    """Maps classes (0,2,4) to indexes (0,1,2)"""
    sub_dict = {0: 0, 2: 1, 4: 2}
    return sub_dict[x]


# Clean tweets
inputs = np.array(tweets_data['Tweet'].apply(lambda x: clean_sentence(x)))
# One-hot-encode classes
targets = tweets_data["Class"].apply(lambda x: substitute_classes(x))
targets = to_categorical(targets, num_classes=3)

# Fit encoder on input data - create vocabulary of NUM_WORDS most frequent words
encoder = create_encoder(inputs)

test_accuracies = []
histories_storage = HistoriesStorage()
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
fold_no = 1

# Perform a model cross-validation for obtaining more reliable results of the model
for train_indexes, test_indexes in kfold.split(inputs, targets):
    clear_session()     # restart Keras global status
    train_inputs, train_targets, test_inputs, test_targets = inputs[train_indexes], targets[train_indexes], \
                                                             inputs[test_indexes], targets[test_indexes]
    model = create_model(encoder)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    history = model.fit(train_inputs, train_targets,
                        batch_size=BATCH_SIZE,
                        epochs=NUMBER_OF_EPOCHS)
    histories_storage.store_history(history)

    train_loss, train_acc = model.evaluate(train_inputs, train_targets, verbose=False)
    print(f'Train Loss: {train_loss}')
    print(f'Train Accuracy: {train_acc}')
    test_loss, test_acc = model.evaluate(test_inputs, test_targets, verbose=False)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc}')
    test_accuracies.append(test_acc)
    fold_no += 1

print(f'Best Test acc: {max(test_accuracies)}')
print(f'Worst Test acc: {min(test_accuracies)}')
print(f'Avg Test acc: {np.mean(test_accuracies)}')

plot_model_histories(histories_storage, 'accuracy')
