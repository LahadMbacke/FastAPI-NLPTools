from fastapi import FastAPI
from fastapi.responses import HTMLResponse  
import ner


app = FastAPI()

@app.get("/ner")
def get_ner(text: str):
    return ner.ner_spacy(text)
    