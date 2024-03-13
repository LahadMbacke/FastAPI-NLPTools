import json
from utils import tokeniser, stemmer_text

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


