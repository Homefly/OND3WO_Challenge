""" test_preprocessing

"""

import unittest, os
import numpy as np
from numpy.testing import *

from preprocessing import parse_training_data, create_datasets, \
    clean_up_sentence, bow


class Preprocessing_Test_Cases(unittest.TestCase):
    def test_parse_training_data(self):
        data_path = os.path.join("data/", "data_intents.json")
        words, classes, documents = parse_training_data(data_path)

        # Types
        self.assertEqual(list, type(words), "Incorrect type of words")
        self.assertEqual(list, type(classes), "Incorrect type of words")
        self.assertEqual(list, type(documents), "Incorrect type of documents")

        for word in words:
            self.assertEqual(str, type(word), "Incorrect type of word in words")
        for class_ in classes:
            self.assertEqual(str, type(class_),
                             "Incorrect type of class in classes")
        for doc in words:
            self.assertEqual(str, type(doc),
                             "Incorrect type of doc in documents")

        # Number of vals
        self.assertEqual(105, len(words), "Incorrect num of words")
        self.assertEqual(9, len(classes), "Incorrect num of classes")
        self.assertEqual(127, len(documents), "Incorrect num of documents")


    def test_create_datasets(self):
        data_path = os.path.join("data/", "data_intents.json")
        words, classes, documents = parse_training_data(data_path)
        (X_train, y_train), (X_test, y_test) = create_datasets(words, classes,
                                                               documents)

        # Types
        self.assertEqual(np.ndarray, type(X_train), "Incorrect type of X_train")
        self.assertEqual(np.ndarray, type(y_train), "Incorrect type of y_train")
        self.assertEqual(np.ndarray, type(X_test), "Incorrect type of X_test")
        self.assertEqual(np.ndarray, type(y_test), "Incorrect type of y_test")

        # Size
        self.assertEqual(114, len(X_train), "Incorrect len of X_train")
        self.assertEqual(114, len(y_train), "Incorrect len of y_train")
        self.assertEqual(13, len(X_test), "Incorrect len of X_test")
        self.assertEqual(13, len(X_test), "Incorrect len of y_test")

        # Values
        for label_vec in y_train:
            self.assertEqual(set([0,1]), set(label_vec),
                             "False val in train label vector")
        for label_vec in y_test:
            self.assertEqual(set([0,1]), set(label_vec),
                             "False val in test label vector")


    def test_clean_up_sentence(self):
        mock_sentence = "This is a mock sentence"
        w_list = clean_up_sentence(mock_sentence)

        self.assertEqual(3, len(w_list),
                         "Incorrect number of words in word list")


    def test_bow(self):
        mock_sentence = "Bye, bye"
        mock_words = ["bye", "donkey"]
        bow_list = bow(mock_sentence, mock_words)

        assert_array_equal(bow_list, np.array([2, 0]),
                           "False BoW implementation")



if __name__ == '__main__':
    unittest.main()
