import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import joblib
import argparse

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['wrist_x', 'wrist_y', 'wrist_z', 'elbow_x', 'elbow_y', 'elbow_z', 'shoulder_x', 'shoulder_y', 'shoulder_z']]
    y = data['label']
    return X, y

def load_model(model_type):
    if model_type == 'nn':
        return tf.keras.models.load_model('./models/model_nn.h5')
    elif model_type == 'gb' or model_type == 'svm':
        return joblib.load(f'./models/model_{model_type}.pkl')
    else:
        raise ValueError("Invalid model type. Choose 'nn' for neural network, 'gb' for gradient boosting, or 'svm' for support vector machine")

def evaluate_model(model, X_test, y_test, model_type):
    if model_type == 'nn':
        y_pred = model.predict(X_test)
        y_pred = np.round(y_pred).astype(int).flatten()
    elif model_type == 'gb' or model_type == 'svm':
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model accuracy.')
    parser.add_argument('model_type', choices=['nn', 'gb', 'svm'], help="Type of model to evaluate: 'nn' for neural network, 'gb' for gradient boosting, 'svm' for support vector machine")
    args = parser.parse_args()

    model_path = f'./models/model_{args.model_type}.h5' if args.model_type == 'nn' else f'./models/model_{args.model_type}.pkl'
    test_data_path = './data/dataset.csv'
    
    model = load_model(args.model_type)
    X_test, y_test = load_data(test_data_path)
    accuracy = evaluate_model(model, X_test, y_test, args.model_type)
    print(f'Model accuracy: {accuracy * 100:.2f}%')