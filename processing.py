import random
import json
import torch
from model import NeuralNetwork
import nltk
from nltk.stem import PorterStemmer
import numpy as np


def request_sentence(sentence):
    bot_name = "robot"
    stemmer = PorterStemmer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('intents_data_source.json', 'r') as json_file:
        intents = json.load(json_file)
    FILE = 'data.pth'
    data = torch.load(FILE)
    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    tags = data['tags']
    model_state = data['model_state']

    model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    sentence = nltk.word_tokenize(sentence)
    tokenized_sentence = [stemmer.stem(w) for w in sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    x = bag
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    confidence = prob.item()
    print(f'Confidence={confidence}')

    if confidence > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return json.dumps({'username': f'{bot_name}',
                'content': random.choice(intent["responses"]),
                'isCurrentUser': False})
    else:
        return json.dumps({'username': f'{bot_name}',
                           'content': "I don't Understand",
                           'isCurrentUser': False})
