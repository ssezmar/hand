import argparse
import numpy as np
import tensorflow as tf
import joblib

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict arm position (bent or extended) based on 3D coordinates.')
    parser.add_argument('model_type', choices=['nn', 'gb', 'svm'], help="Type of model to use: 'nn' for neural network, 'gb' for gradient boosting, 'svm' for support vector machine")
    parser.add_argument('points', metavar='N', type=float, nargs=9,
                        help='9 float values representing the 3D coordinates of shoulder, elbow, and wrist (x, y, z for each).')
    args = parser.parse_args()
    
    model = load_model(args.model_type)
    result = predict_arm_position(model, args.points, args.model_type)
    print(f'Prediction: {"Bent" if result == 1 else "Extended"}')