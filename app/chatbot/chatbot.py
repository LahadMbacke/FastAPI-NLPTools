import random
import json
import os
import torch
from utils import tokeniser, stemmer_text, bag_of_words
from model_chatbot import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.expanduser('~/Documents/FastAPI-NLPTools/data/intents.json'), 'r') as f:
    intents = json.load(f)

FILE = os.path.expanduser('~/Documents/FastAPI-NLPTools/data/data.pth')
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

model_state = data['model_state']
model = NeuralNet(input_size, hidden_size, output_size).to(device) 
model.load_state_dict(model_state) # chargement du modèle
model.eval() # mode évaluation

bot_name = "Lahad"
print("Start of chatting! (type 'quit' to exit)")

while True:
    phrases = input("You: ")
    if phrases == "quit":
        break

    phrase = tokeniser(phrases)
    phrase = stemmer_text(phrase)
    X = bag_of_words(phrase, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")