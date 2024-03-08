import spacy
from spacy import displacy


nlp = spacy.load("fr_core_news_md")
def ner_spacy(text):
    doc = nlp(text)
    return displacy.render(doc, style="ent", page=True)