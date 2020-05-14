from flask import Flask
from flask import request
from util import get_prediction
from flask.json import jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.json['tweet']
        result = get_prediction(tweet)
    return jsonify({'tweet': tweet, 'status':result})