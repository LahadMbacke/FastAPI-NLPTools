from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
def test():
    # return "<h1>Test</h1>"
    return {"message": "Hello World"}