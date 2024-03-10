from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import ner
from pydantic import BaseModel



class Text(BaseModel):
    text: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, text: str = ""):
    return templates.TemplateResponse("ner.html", {"request": request})


@app.post("/ner")
async def ner_api(text: Text):
    return ner.ner_spacy(text.text)