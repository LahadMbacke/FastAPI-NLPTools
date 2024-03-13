import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


def tokeniser(text):
    return nltk.word_tokenize(text)
def stemmer(text):
    return [stemmer.stem(w.lower()) for w in text]

def bag_of_words(text):
    pass

