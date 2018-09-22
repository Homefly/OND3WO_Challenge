""" chatbot

Conversation logic and helpers of Chatbot

"""

import os, sys, json, random
import numpy as np
from preprocessing import parse_training_data, bow


def classify(sentence, model):
    """ Make predictions

    :param sentence: pattern
    :param model: trained model
    :return: [(intent, proba)]
    """
    ERROR_THRESHOLD = 0.0001
    data_path = os.path.join("data/", "data_intents.json")
    words, classes, documents = parse_training_data(data_path)

    # generate probabilities from the model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, model, user_id='123', context={}, show_details=False):
    """ Generates a contextualized response for a specific user

    :param sentence: pattern
    :param model: trained model
    :param user_id: user id
    :param show_details:
    :return: -
    """
    # Load intents
    data_path = os.path.join("data/", "data_intents.json")
    with open(data_path) as json_data:
        intents = json.load(json_data)

    # Classify sentence
    results = classify(sentence, model)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print('context:', i['context_set'])
                        context[user_id] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        if i["tag"] == "goodbye":
                            print(random.choice(i['responses']))
                            sys.exit()
                        else:
                            return print(random.choice(i['responses']))

            results.pop(0)


def conversation(model):
    """ Conversation with bot

    :param model: fitted ffNN
    :return: -
    """

    print("To exit, say goodbye ('Bye', 'Ciao', 'See you', etc.) or kill the "
          "conversation with Keyboard interrupt (CTRL+C)")
    print("Let's start the conversation...")

    context = {}
    while True:
        q = input("Q: ")
        response(q, model, "datadonk23", context=context)
