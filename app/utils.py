import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


def tokeniser(text):
    return nltk.word_tokenize(text)
def stemmer_text(text):
    ignore_words = ['?', '!', '.', ',']
    return [stemmer.stem(w.lower()) for w in text if w not in ignore_words]

def bag_of_words(tokenised_phr, all_words):
    tokenised_phr = [stemmer.stem(w) for w in tokenised_phr ] 
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenised_phr:
            bag[idx] = 1.0
    return bag


words = ["hello","I","you","bye","thank","cool"]
sentence = ["hello","how","are","you"]
bag = bag_of_words(sentence, words)
print(bag)



