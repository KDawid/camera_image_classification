from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify
import json
import numpy as np
from sklearn.svm import LinearSVC
import tensorflow as tf

from camera import PiCamera
import lenet
from predict_images import ImagePredictor
from train_model import ModelTrainer

app = Flask(__name__)
api = Api(app)


graph = tf.Graph()
with graph.as_default():
    session = tf.Session()
    with session.as_default():
        predictor = ImagePredictor('training', predictions_dir='predictions')


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/predict', methods=['GET'])
def predict():
    with graph.as_default():
        with session.as_default():
            prediction = predictor.predict_image()
    result = {"prediction": prediction}
    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict_image():
    data = request.get_json()
    img = np.array(data)
    with graph.as_default():
        with session.as_default():
            prediction = predictor.predict(img)
    result = {"prediction": prediction}
    return jsonify(result)


@app.route('/image')
def get_image():
    img = predictor.camera.get_img_array()
    result = {"image": img.tolist()}
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')
