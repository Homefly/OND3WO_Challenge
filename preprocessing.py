""" preprocessing

NLP tasks

"""

import json, nltk, random, pickle, string
random.seed(23)
nltk.data.path.append("/home/<user>/nltk_data")
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def clean_sequence(seq):
    """ Stemming, remove stopwords and punctuation from input sequence

    :param seq: [word]
    :return: [word] - cleaned seq of words
    """
    stemmer = LancasterStemmer()
    with open("data/custom_stopwords.p", "rb") as f:
        stop_words = pickle.load(f)

    cleaned_seq = [stemmer.stem(w.lower()) for w in seq if w not in
                   stop_words and w not in string.punctuation]

    return cleaned_seq


def parse_training_data(path):
    """ Parse training data and create dictonary for app

    :param path: path to training data
    :return: [word], [class], [document]
    """
    words = []
    classes = []
    documents = []

    with open(path) as json_data:
        intents = json.load(json_data)

    # loop through each sentence in our intents patterns
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:

            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)

            # add to our words list
            words.extend(w)

            # add to documents in our corpus
            documents.append((w, intent["tag"]))

            # add to our classes list
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    # stem and lower each word and remove duplicates
    words = clean_sequence(words)
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    return words, classes, documents


def create_datasets(words, classes, documents):
    """ Create train- & testset

    :param words: list of parsed words
    :param classes: list of parsed classes
    :param documents: list of parsed docs
    :return: Trainset, Testset
    """
    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = [0] * len(words) # BUGFIX
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = clean_sequence(pattern_words)
        # create our bag of words array - BUGFIX
        for pw in pattern_words:
            for i, w in enumerate(words):
                if w == pw:
                    bag[i] += 1

        # class index as label
        target_num = doc[1]

        training.append([bag, target_num])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists, dirty hack because of keras input specifics
    X = np.vstack(training[:, 0])
    y = training[:, 1]
    y = pd.get_dummies(y)
    y = y.values.argmax(1)
    y = to_categorical(y, len(classes))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=23, shuffle=True)

    return (X_train, y_train), (X_test, y_test)


def clean_up_sentence(sentence):
    """ Split sentence into clean list of words

    :param sentence: pattern
    :return: [word]
    """
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = clean_sequence(sentence_words)
    return sentence_words


def bow(sentence, words, show_details=False):
    """ Creates BoW

    :param sentence: pattern
    :param words: words list
    :param show_details:
    :return:
    """
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] += 1
                if show_details:
                    print ("found in bag: %s" % w)

    return (np.array(bag))