""" test_chatbot

"""

import unittest
from keras.models import load_model

from chatbot import classify

class ChatbotTestCases(unittest.TestCase):

    def test_classify(self):
        model = load_model("model/model_ffNN.best.hdf5")
        results = classify("Are you open today?", model)

        self.assertEqual(9, len(results), "Incorrect len of classifications")
        self.assertEqual(2, len(results[0]), "Fals len of classification tuple")
        self.assertEqual("opentoday", results[0][0], "Incorrect intent "
                                                     "classified")
        self.assertTrue(results[0][1] <= 1., "Classified probe too high")
        self.assertTrue(results[0][1] > 0., "Classified probe too low")


if __name__ == '__main__':
    unittest.main()
