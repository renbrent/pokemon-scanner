from flask import Flask, jsonify
from model.model import get_predictions
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=["GET"])
def predict():
    return jsonify(get_predictions('../model/dataset',
                           '../model/test_dataset',
                           '../model/model_weights.pth'))

if __name__ == '__main__':
    app.run()
