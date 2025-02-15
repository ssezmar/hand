import argparse
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model(model_type):
    if model_type == 'nn':
        return tf.keras.models.load_model('./models/model_nn.h5')
    elif model_type == 'gb' or model_type == 'svm':
        return joblib.load(f'./models/model_{model_type}.pkl')
    else:
        raise ValueError("Invalid model type. Choose 'nn' for neural network, 'gb' for gradient boosting, or 'svm' for support vector machine")

def predict_arm_position(model, points, model_type):
    points = np.array(points).reshape(1, -1)
    if model_type == 'nn':
        prediction = model.predict(points)
        return int(np.round(prediction[0][0]))
    elif model_type == 'gb' or model_type == 'svm':
        prediction = model.predict(points)
        return int(prediction[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.get('model_type')
    points = data.get('points')
    
    if not model_type or not points:
        return jsonify({'error': 'Invalid input'}), 400
    
    model = load_model(model_type)
    result = predict_arm_position(model, points, model_type)
    response = jsonify({'prediction': 'Согнута' if result == 1 else 'Разогнута'})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the prediction server.')
    parser.add_argument('model_type', choices=['nn', 'gb', 'svm'], help="Type of model to load: 'nn' for neural network, 'gb' for gradient boosting, 'svm' for support vector machine")
    args = parser.parse_args()
    
    model = load_model(args.model_type)
    app.run(host='0.0.0.0', port=5000)