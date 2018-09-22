""" stopword_utils

Compuile a combined stopwords list

"""

import nltk, pickle
nltk.data.path.append("/home/<user>/nltk_data")
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS

custom_stopwords = []

nltk_sw = stopwords.words("english")
sklearn_sw = list(stop_words.ENGLISH_STOP_WORDS)
spacy_sw = list(STOP_WORDS)

combined_sw = set(sklearn_sw + nltk_sw + spacy_sw)

# Words, which shouldn't be removed to provide personal or
# conversational context and understanding of commands
commands = ["yes", "no", "copied", "yeah", "on", "off", "quit", "end"]
personal = ["me", "i", "we", "us", "all"]
q_words = ["who", "when", "what", "why", "which", "where", "how"]
chatbot_specific = ["bye", "thanks", "rent", "open", "today"]

stopwords_to_remove = commands + personal + q_words + chatbot_specific

for w in combined_sw:
    if w not in stopwords_to_remove:
        custom_stopwords.append(w)

pickle.dump(custom_stopwords, open("../data/custom_stopwords.p", "wb"))
