import json
import os
import numpy as np
from utils import tokeniser, stemmer_text, bag_of_words
import torch
from torch.utils.data import Dataset, DataLoader
from model_chatbot import NeuralNet

with open(os.path.expanduser('~/Documents/FastAPI-NLPTools/data/intents.json'), 'r') as f:    
    data = json.load(f)

all_words = []
tags = []
word_tag_pairs = []

for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokeniser(pattern)
        all_words.extend(word)
        word_tag_pairs.append((word, tag))

all_words = stemmer_text(all_words)
# all_words = [sorted(set(all_words))]
# tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern, tag) in word_tag_pairs:
    bag = bag_of_words(pattern, all_words) # bag is a list
    X_train.append(bag) # X_train is a list of lists
    label = tags.index(tag)
    y_train.append(label) # y_train is a list of integers

X_train = np.array(X_train)
y_train = np.array(y_train)

class chatBotDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
dataset = chatBotDataset()  # instance de la classe chatBotDataset
train_loader = DataLoader(dataset=dataset,
                          batch_size=8,
                          shuffle=True,
                          num_workers=2)

hidden_size = 8  # nombre de neurones dans la couche cachée
output_size = len(tags) # nombre de classes
input_size = len(all_words) # taille du vecteur d'entrée
learning_rate = 0.001

# print(input_size, len(X_train[0]))
# print(output_size, tags)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device) # instance de la classe NeuralNet

criterion = torch.nn.CrossEntropyLoss() # fonction de coût pour la classification multiclasse 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimiseur Adam pour l'entraînement du modèle 

num_epochs = 1000
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels.long()) # calcul de la fonction de coût

        # Backward and optimize
        optimizer.zero_grad() # remise à zéro des gradients
        loss.backward() # rétropropagation
        optimizer.step() # mise à jour des paramètres

    if (epoch+1) % 100 == 0: # affichage de la fonction de coût tous les 100 epochsclear
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(), # état du modèle
    "input_size": input_size, # taille du vecteur d'entrée
    "output_size": output_size, # nombre de classes
    "hidden_size": hidden_size, # nombre de neurones dans la couche cachée
    "all_words": all_words, 
    "tags": tags
}
FILE = os.path.expanduser('~/Documents/FastAPI-NLPTools/data/data.pth')
torch.save(data, FILE)
print(f'Entraînement terminé. Modèle sauvegardé sous {FILE}')
