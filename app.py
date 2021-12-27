from flask import Flask, request
from processing import request_sentence
import time
app = Flask(__name__)


@app.route('/', methods=['POST'])
def get_bot_response():
    if request.method == 'POST':
        post_sentence = request.get_json()['content']
        print(post_sentence)
        confidence_sentence = request_sentence(post_sentence)
        time.sleep(1)
        return confidence_sentence


if __name__ == '__main__':
    app.run()
