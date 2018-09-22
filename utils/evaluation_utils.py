""" evaluation_utils

"""

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def eval_clf(y_test, y_pred, train_acc):
    """ Evaluate classifier on specific metrices

    :param y_test: true vals
    :param y_pred: predicted vals
    :param train_acc: train accuracy (pre-computed)
    :return: [test_acc, precision, recall, fscore, train_acc]
    """
    train_acc = train_acc.round(3)
    scores = precision_recall_fscore_support(y_test, y_pred, average="macro")
    precision, recall, fscore = scores[0].round(3), scores[1].round(3), \
                                scores[2].round(3)
    test_acc = accuracy_score(y_test, y_pred).round(3)

    return [test_acc, precision, recall, fscore, train_acc]
