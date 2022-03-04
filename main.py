import torch
import torch.nn as nn
from preprocessing import PreProcessing
from torch.utils.data import DataLoader
from chat_dataset import ChatDataSet
from dotenv import load_dotenv
from model import NeuralNetwork
import os.path
import ssl
import nltk

load_dotenv = load_dotenv()
if not os.path.exists(os.getenv('NLTK_PATH')):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("punkt")

preprocessing = PreProcessing()
preprocessing.load_json('intents_data_source.json')
preprocessing.tokenization()
preprocessing.stem_and_clean_data()
preprocessing.bag_of_word()


# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(preprocessing.x_train[0])
hidden_size = 8
output_size = len(preprocessing.tags)


dataset = ChatDataSet(x_train=preprocessing.x_train, y_train=preprocessing.y_train)

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)

# loss and optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        # backwards and optimizers step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch{epoch+1}/{num_epochs}, loss={loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": preprocessing.all_words,
    "tags": preprocessing.tags
}

File = "data.pth"
torch.save(data, File)
