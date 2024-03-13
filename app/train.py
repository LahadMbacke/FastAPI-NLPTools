import json
import numpy as np
from utils import tokeniser, stemmer_text, bag_of_words

with open('../data/intents.json', 'r') as f:
    data = json.load(f)

all_words = []
tags = []
word_tag_pairs = []

for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokeniser(pattern)
        all_words.extend(word)
        word_tag_pairs.append((word, tag))


all_words = [sorted(set(all_words))]
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern, tag) in word_tag_pairs:
    bag = bag_of_words(pattern, all_words) # TODO: Implement bag_of_words function
    X_train.append(bag) # X_train is a list of lists
    label = tags.index(tag)
    y_train.append(label) # y_train is a list of integers

X_train = np.array(X_train)
y_train = np.array(y_train)