""" evaluation

Model evaluation

"""

from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

from utils.tensorboard_utils import make_callbacks
from utils.evaluation_utils import eval_clf

from models import feedforward_models, modelInputWidth, convolutional_models


def evaluate_models(trainset, testset, model_names, trained_models):
    """ Evaluate models and plot test summary

    :param trainset: train data
    :param testset: test data
    :param model_names: [model_name]
    :param trained_models: [trained_model]
    :return: -
    """
    (X_train, y_train) = trainset
    (X_test, y_test) = testset
    eval_results = {}

    # Input reshaping
    X_train_rec = X_train.reshape(114, 1, modelInputWidth)
    X_test_rec = X_test.reshape(13, 1, modelInputWidth)
    y_train_ensemble = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    for name, model in zip(model_names, trained_models):
        if name in ["LSTM", "BiLSTM"]:
            train_acc = model.score(X_train_rec, y_train)
            y_pred = model.predict(X_test_rec)
        elif (name == "ffNN"):
            train_acc = model.score(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            train_acc = model.score(X_train, y_train_ensemble)
            y_pred = model.predict(X_test)

        eval_results[name] = eval_clf(y_test, y_pred, train_acc)

    evalDF = pd.DataFrame.from_dict(eval_results, orient="index",
                                  columns=["Test_acc", "Precision",
                                           "Recall", "FBeta",
                                           "Train_acc"]).sort_values(
        ["Test_acc", "Train_acc"], ascending=False)
    evalDF.to_csv("model/test_summary.csv")
    print(evalDF)


class ModelExplorer:
    """ Helper class for model selection

    Template by David Batista, Panagiotis Katsaroumpas
    (http://www.davidsbatista.net/blog/2018/02/23/model_optimization/)
    """
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by="mean_score"):
        def row(key, scores, params):
            d = {
                "estimator": key,
                #"min_score": min(scores),
                "mean_score": np.mean(scores),
                "max_score": max(scores),
                #"std_score": np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            #print(k)
            params = self.grid_searches[k].cv_results_["params"]
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by],
                                                      ascending=False)

        columns = ["estimator", "mean_score", "max_score"]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


def explore_models(trainset):
    """ Explore models to find best parameters. Plots summaries in model/

    :param trainset: (X_train, y_train)
    :return: -
    """
    (X_train, y_train) = trainset

    # ffNN
    models, params = feedforward_models()
    ff_NNs = ModelExplorer(models, params)
    ff_NNs.fit(X_train, y_train, cv=5, n_jobs=-1)
    summary = ff_NNs.score_summary(sort_by="mean_score")
    summary.to_csv("model/ffNNs_summary.csv")

    # CNN
    models, params = convolutional_models() 
    c_NNs = ModelExplorer(models, params)
    X_train = np.expand_dims(X_train, axis=2)
    c_NNs.fit(X_train, y_train, cv=5, n_jobs=-1)
    summary = c_NNs.score_summary(sort_by="mean_score")
    summary.to_csv("model/cNNs_summary.csv")

    #LSTM


    #DNN
