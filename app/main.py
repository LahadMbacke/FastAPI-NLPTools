from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app import ner


app = FastAPI()

# Routes pour NER
app.include_router(ner.router, prefix="/ner", tags=["NER"])
