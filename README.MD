# CHATBOT API

These are based on NLP. Natural language processing for chatbot makes such bots very human-like. They react to the meaning of the whole question. The AI-based chatbot can learn from every interaction and expand their knowledge.sing scrapping (free API).<br>

## Installation

Install dependencies

```bash
pip install -r requirements.txt
```
If you get errors installing dependencies, use Python 3.9.6.

Create a file ```.env``` with "nltk_data" location path.
example: ```NLTK_PATH=/Users/yourusername/nltk_data```
if this file does not exist it will be created
## Information
NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

## Usage

Run ```main.py``` to start machine learning preprocessing from the ```intents_data_source.json``` file

```data.pth``` file will be create.

Now we can start Flask server ```app.pth```

```python
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
```

Send message with POST Method 

![alt text](https://res.cloudinary.com/dhdgnx4mc/image/upload/v1646432443/media/GitHub/gm3v2ifjr3ee8gbsp45y.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.