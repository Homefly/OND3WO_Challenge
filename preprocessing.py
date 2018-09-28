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
from textblob import TextBlob

class Preprocessing(object):
    
    def clean_up_sentence(self, sentence):
        """ Split sentence into clean list of words

        :param sentence: pattern
        :return: [word, word,...]
        """
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words =self._clean_sequence(sentence_words)
        return sentence_words

    def _check_pos_tag(sent, flag):
        """ Function to check and get the part of speech tag count of a words in a given sentence
        
        :param sent: sentance
        :param flag: part of speach 
        :return: count of words in a specific part of speach
        """

        #Parts of Speach
        pos_family = {
            'noun' : ['NN','NNS','NNP','NNPS'],
            'pron' : ['PRP','PRP$','WP','WP$'],
            'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
            'adj' :  ['JJ','JJR','JJS'],
            'adv' : ['RB','RBR','RBS','WRB']
        }

        cnt = 0
        for word in sent:
            try:
                wiki = TextBlob(word)
                for tup in wiki.tags:
                    ppo = list(tup)[1]
                    if ppo in pos_family[flag]:
                        cnt += 1
            except:
                pass
        return cnt

    def _clean_sequence(seq):
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

    @classmethod
    def parse_training_data(cls, path):
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
        words = cls._clean_sequence(words)
        words = sorted(list(set(words)))

        # remove duplicates
        classes = sorted(list(set(classes)))

        return words, classes, documents

    def create_BOW_array(words, classes, documents):
        """ Creates bag of words vectors and tags

        :param word: words used for BOW vector
        :param classes: tags, catagories of patterns
        :param documents: parced + stemed sentances with tags
        :return: BOW vector with tags
        """
        # create our training data
        trainingBOW = []
        # create an empty array for our output
        output_empty = [0] * len(classes)

        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = [0] * len(words) # BUGFIX
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = Preprocessing._clean_sequence(pattern_words)
            # create our bag of words array - BUGFIX
            for pw in pattern_words:
                for i, w in enumerate(words):
                    if w == pw:
                        bag[i] += 1

            # class index as label
            target_num = doc[1]

            trainingBOW.append([bag, target_num])
        return trainingBOW

    @classmethod
    def create_datasets(cls, words, classes, documents):
        """ Create train- & testset

        :param words: list of parsed words
        :param classes: list of parsed classes
        :param documents: list of parsed docs
        :return: Trainset, Testset
        """

        # get BOW Vecotrs
        trainingBOW = cls.create_BOW_array(words, classes, documents)
        tags = [tag[1] for tag in trainingBOW]
        trainingBOW = [vec[0] for vec in trainingBOW]

        trainingAdv = cls.create_advanced_feat_training_data(classes, documents)

        #BOW vector combined with advanced attributes
        trainingTot = [x + y for x, y in zip(trainingBOW, trainingAdv)]
        trainingTot = [*zip(trainingTot, tags)]

        # shuffle our features and turn into np.array
        random.shuffle(trainingTot)
        trainingTot = np.array(trainingTot)

        # create train and test lists, dirty hack because of keras input specifics
        X = np.vstack(trainingTot[:, 0])
        #X = X[:, 0:105] #This gives BOW elements of vectro only
        y = trainingTot[:, 1]
        y = pd.get_dummies(y)
        y = y.values.argmax(1)
        y = to_categorical(y, len(classes))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=23, shuffle=True)

        #import ipdb; ipdb.set_trace()
        return (X_train, y_train), (X_test, y_test)

    def create_advanced_feat_training_data(classes, documents):
        """ Creates advanced feature vectors

        :param classes: tags, catagories of patterns
        :param documents: list of parsed docs
        :return: list of advanced feature vectors
        """
        advFeatDF = Preprocessing.additional_features(classes, documents)

        advFeatures = ['char_count', 'word_count', 'word_density',
       'punctuation_count', 'title_word_count', 'uppercase_word_count',
       'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']        
        
        advFeatArrayNorm = pd.DataFrame()
        for feature in advFeatures:
            advFeatArrayNorm[feature] = Preprocessing.normalize_DF_column(advFeatDF, feature)

        return advFeatArrayNorm.values.tolist()

    def normalize_DF_column(df, column):
        return (df[column] - df[column].min())/(df[column].max() - df[column].min())

    @classmethod
    def additional_features(cls, classes, documents):
        """ Calculates advanced feature values
        :param classes: list of parsed classes
        :param documents: list of parsed docs
        :return: Dataframe with: Word Count, 
                                Character Count,
                                Average Word Density,
                                Puncutation Count,
                                Upper Case Count,
                                Title Word Count,
                                Part of Speech Freq
        """
        
        trainDF = pd.DataFrame()
        trainDF['patterns'] = [pattern[0] for pattern in documents]
        trainDF['tag']      = [tag[1] for tag in documents]

        #Counts:do not include white space charcters
        trainDF['char_count'] = trainDF['patterns'].apply(lambda x: sum([*map(len, x)]))
        trainDF['word_count'] = trainDF['patterns'].apply(len)
        trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count'])
        
        #Punctuation count: number of punctuation marks in pattern
        puncs_in_string = lambda x: sum([1 for _ in list(x) if _ in string.punctuation])
        trainDF['punctuation_count'] = trainDF['patterns'].apply(lambda x: sum([*map(puncs_in_string, x)])) 

        #Title Word Count: number of words starting with uppercase letter
        trainDF['title_word_count'] = trainDF['patterns'].apply(lambda x: sum([*map(str.istitle, x)]))

        #Uppercase word count: number of words in all upper case 
        trainDF['uppercase_word_count'] = trainDF['patterns'].apply(lambda x: sum([*map(str.isupper, x)]))

        trainDF['noun_count'] = trainDF['patterns'].apply(lambda x:cls._check_pos_tag(x, 'noun'))
        trainDF['verb_count'] = trainDF['patterns'].apply(lambda x:cls._check_pos_tag(x, 'verb'))
        trainDF['adj_count']  = trainDF['patterns'].apply(lambda x:cls._check_pos_tag(x, 'adj' ))
        trainDF['adv_count']  = trainDF['patterns'].apply(lambda x:cls._check_pos_tag(x, 'adv' ))
        trainDF['pron_count'] = trainDF['patterns'].apply(lambda x:cls._check_pos_tag(x, 'pron'))
        
        return trainDF

    def bow(cls, sentence, words, show_details=False):
        """ Creates BoW

        :param sentence: pattern
        :param words: words list
        :param show_details:
        :return: single BoW vec
        """
        # tokenize the pattern
        sentence_words = cls.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s:
                    bag[i] += 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return (np.array(bag))
