""" models
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier

#from evaluation import *
modelInputWidth = 116

def create_ffNN(lr=0.005, decay=0.001):
    """ Create Feed-Forward NN, from template

    :param lr: learning rate for adam optimizer
    :param decay: learning rate decay for adam optimizer
    :return: model
    """
    model = Sequential()
    model.add(Dense(100, activation="relu", input_dim=modelInputWidth))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(9, activation="softmax"))

    # metrics
    adam = optimizers.Adam(lr=lr, decay=decay)
    model.compile(loss="categorical_crossentropy",
                  optimizer=adam,
                  metrics=["accuracy"])
    # print(model.summary())

    return model


def create_LSTM(optimizer="adam"):
    """ Create simple LSTM model

    :param optimizer: keras optimizer
    :return: model
    """
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, modelInputWidth)))
    model.add(Dropout(0.1))
    model.add(Dense(9, activation="softmax"))

    # metrics
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    # print(model.summary())

    return model


def create_biLSTM(optimizer="adam"):
    """ Create simple bidirectional LSTM model

    :param optimizer: keras optimizer
    :return: model
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=(1, modelInputWidth))))
    model.add(Dropout(0.1))
    model.add(Dense(9, activation="softmax"))

    # metrics
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    # print(model.summary())

    return model


def feedforward_models():
    """ Creates dict of feed-forward NN models and params

    :return: ({"name": clf}, {"name", {"param": value_space}})
    """
    models = {
        "ffNN": KerasClassifier(build_fn=create_ffNN,
                                batch_size=32, verbose=1, validation_split=0.2)
    }

    params = {
        "ffNN": {"epochs": [35, 40, 45, 80],
                 "lr": [0.005, 0.01],
                 "decay": [0, 0.001]}
    }

    return (models, params)


def recurrent_models():
    """ Creates dict of recurrent NN models and params

    :return: ({"name": clf}, {"name", {"param": value_space}})
    """
    models = {
        "LSTM": KerasClassifier(build_fn=create_LSTM, batch_size=32,
                                verbose=1, validation_split=0.2),
        "biLSTM": KerasClassifier(build_fn=create_biLSTM, batch_size=32,
                                  verbose=1, validation_split=0.2)
    }

    params = {
        "LSTM": {"epochs": [120, 150, 180],
                 "optimizer": ["RMSProp", "adam"]},
        "biLSTM": {"epochs": [100, 150, 180],
                   "optimizer": ["RMSProp", "adam"]}
    }

    return (models, params)


def ensemble_models():
    """ Creates dict of ensemble models and params

    :return: ({"name": clf}, {"name", {"param": value_space}})
    """
    models = {
        "Bagging": BaggingClassifier(random_state=23),
        "RandomForest": RandomForestClassifier(random_state=23)
    }

    params = {
        "Bagging": {"n_estimators": [15, 30], "max_features": [
            1., 0.9, 0.8], "n_jobs": [1]},
        "RandomForest": {"n_estimators": [10, 20],
                         "criterion": ["gini", "entropy"],
                         "max_features": ["sqrt", "log2"],
                         "n_jobs": [-1]}
    }

    return (models, params)


def best_ensemble():
    """ Ensemble models with best params taken from
    model/ensembleCLFs_summary.csv

    :return: zip("name", model)
    """
    names = ["Bagging", "RandomForest", "ExtraTrees", "AdaBoost",
             "GradBoost", "Voting"]

    # Params taken from model/ensembleCLF.csv
    bagging_clf = BaggingClassifier(n_estimators=30, max_features=0.8,
                                    random_state=23)
    rf_clf = RandomForestClassifier(n_estimators=20, max_features="log2",
                                    criterion="entropy", random_state=23)

    return zip(names, [bagging_clf, rf_clf])


def stacking():
    """ Stack best models into one CLF

    :return: stacked_model
    """
    classifiers = []

    # List of CLFs
    for (_, model) in best_ensemble():
        classifiers.append(model)
    # Meta-learner
    superlearner = RandomForestClassifier(random_state=23)

    # Stack models
    np.random.seed(23)
    stacking_clf = StackingCVClassifier(classifiers,
                                        meta_classifier=superlearner, cv=5)

    return stacking_clf
