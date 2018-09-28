""" training

Model training

"""

import pickle
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier

from models import create_ffNN, best_ensemble, stacking, modelInputWidth


def train_best_models(trainset):
    """ Train models with best parameters. Returns list of trained models.

    :param trainset: train data
    :return: ([model_name], [trained_model])
    """
    (X_train, y_train) = trainset
    model_names, model_list = [], []

    # Input reshaping
    X_train_rec = X_train.reshape(114, 1, modelInputWidth)
    y_train_ensemble = np.argmax(y_train, axis=1)

    # ffNN
    ffNN = KerasClassifier(build_fn=create_ffNN, batch_size=32,
                           verbose=0, validation_split=0.2)
    ffNN.fit(X_train, y_train, epochs=80)
    model_names.append("ffNN")
    model_list.append(ffNN)

    # Ensemble
    for (name, model) in best_ensemble():
       clf = model.fit(X_train, y_train_ensemble)
       path = "model/model_"+ name + ".best.p"
       with open(path, "wb") as f:
           pickle.dump(clf, f)
       model_names.append(name)
       model_list.append(clf)

    # Stacking
    stacked = stacking().fit(X_train, y_train_ensemble)
    model_names.append("StackedCLF")
    model_list.append(stacked)

    return model_names, model_list
