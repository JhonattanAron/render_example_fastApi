# server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from starlette.middleware.cors import CORSMiddleware


app = FastAPI()

# Carga el modelo y el tokenizador
model = AutoModelForSequenceClassification.from_pretrained("aronDFarkl/csaron-es")
tokenizer = AutoTokenizer.from_pretrained("aronDFarkl/csaron-es")

class InputText(BaseModel):
    text: str


# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes de todos los orígenes (para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

@app.post("/predict")
def predict(input: InputText):
    # Tokeniza el texto
    inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    return {"label": predicted_class}
