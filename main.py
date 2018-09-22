""" main

Chatbot main logic

"""

import os
from keras import backend

from preprocessing import parse_training_data, create_datasets
from models import create_ffNN
from training import train_best_models
from evaluation import explore_models, evaluate_models
from chatbot import conversation


def chat_with_bot():
    """ Starts conversation with bot

    :return: -
    """
    print("Wanna chat with bot Shorty? (ffNN only)")
    input_str = input("[y/n] ")
    if input_str.lower().strip() in ["y", "yes"]:
        shorty = create_ffNN()
        shorty.fit(X_train, y_train, epochs=80, batch_size=32,
                   validation_split=0.2, verbose=0)
        conversation(shorty)
    else:
        print("CU, bye!")


def main():
    """ Main logic

    :return: -
    """
    backend.clear_session()
    global X_train, y_train, X_test, y_test

    # Parse and clean training data
    data_path = os.path.join("data/", "data_intents.json")
    words, classes, documents = parse_training_data(data_path)
    (X_train, y_train), (X_test, y_test) = create_datasets(words, classes,
                                                           documents)

    # Model exploration
    explore_models((X_train, y_train))

    # Train models with best params
    names, models = train_best_models((X_train, y_train))

    # Evaluate models
    evaluate_models((X_train, y_train), (X_test, y_test), names, models)

    # Have a chat with bot, if you'd like
    print()
    chat_with_bot()


if __name__ == '__main__':
    main()