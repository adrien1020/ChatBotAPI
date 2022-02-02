import json
import nltk
from nltk.stem import PorterStemmer
import numpy as np


class PreProcessing(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.ignore_word = ['?', ' ', '!', ',', ';', '.', ':', '/', '#', '@',"'"]
        self.intents = []
        self.all_words = []
        self.tags = []
        self.xy = []
        self.bag = []
        self.x_train = []
        self.y_train = []

    def load_json(self, source):
        with open(source, 'r') as json_file:
            self.intents = json.load(json_file)

    def tokenization(self):
        for intent in self.intents['intents']:
            tag = intent['tag']
            self.tags.append(tag)
            patterns = intent['patterns']
            for pattern in patterns:
                words = nltk.word_tokenize(pattern)
                self.all_words.extend(words)
                self.xy.append((words, tag))

    def stem_and_clean_data(self):
        self.all_words = [self.stemmer.stem(word.lower()) for word in self.all_words if word not in self.ignore_word]
        # delete duplicate in all_world
        self.all_words = sorted(set(self.all_words))
        # delete duplicate in tags
        self.tags = sorted(set(self.tags))

    def bag_of_word(self):
        for (pattern_sentence, tag) in self.xy:
            tokenized_sentence = [self.stemmer.stem(word.lower()) for word in pattern_sentence]
            self.bag = np.zeros(len(self.all_words), dtype=np.float32)
            for idx, word in enumerate(self.all_words):
                if word in tokenized_sentence:
                    self.bag[idx] = 1.0
            self.x_train.append(self.bag)
            label = self.tags.index(tag)
            self.y_train.append(label)
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

