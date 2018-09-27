""" test_training

"""

import unittest, os

from preprocessing import Preprocessing
from training import train_best_models


class TrainingTestCase(unittest.TestCase):
    def test_train_best_models(self):
        data_path = os.path.join("data/", "data_intents.json")
        words, classes, documents = Preprocessing.parse_training_data(data_path)
        (X_train, y_train), _ = Preprocessing.create_datasets(words, classes, documents)
        import ipdb; ipdb.set_trace()
        mock_models = train_best_models((X_train, y_train))

        self.assertEqual(2, len(mock_models), "Returns false tuple len")
        names = mock_models[0]
        models = mock_models[1]
        self.assertTrue(len(names) == len(models), "Length of names and "
                                                   "models list differ")
        self.assertEqual(10, len(models), "False num of models in list")

if __name__ == '__main__':
    unittest.main()
