from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import ner

app = FastAPI()
templates = Jinja2Templates(directory="templates")
@app.get("/ner", response_class=HTMLResponse)
async def get_ner(request: Request, text: str = ""):
    html_content = ner.ner_spacy(text)
    return templates.TemplateResponse("ner.html", {"request": request, "html_content": html_content})

# Votre fonction ner_spacy reste la même
