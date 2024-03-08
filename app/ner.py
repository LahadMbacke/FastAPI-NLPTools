import spacy
from spacy import displacy


nlp = spacy.load("fr_core_news_md")
text = input("Entrez un texte: ")
doc = nlp(text)
print([(X.text, X.label_) for X in doc.ents])