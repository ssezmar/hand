import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from models.model import NeuralNetwork
import joblib
import argparse

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['wrist_x', 'wrist_y', 'wrist_z', 'elbow_x', 'elbow_y', 'elbow_z', 'shoulder_x', 'shoulder_y', 'shoulder_z']]
    y = data['label']
    return X, y

def train_neural_network(X_train, y_train, X_val, y_val):
    input_shape = (X_train.shape[1],)
    num_classes = 1
    model = NeuralNetwork(input_shape, num_classes)
    model.build_model()
    model.train(X_train, y_train, X_val, y_val)
    model.save('../models/model_nn.h5')

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/model_gb.pkl')

def train_svm(X_train, y_train):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, '../models/model_svm.pkl')

def main(model_type):
    # Load the dataset
    X, y = load_data('../data/processed/processed_dataset.csv')
    
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'nn':
        train_neural_network(X_train, y_train, X_val, y_val)
    elif model_type == 'gb':
        train_gradient_boosting(X_train, y_train)
    elif model_type == 'svm':
        train_svm(X_train, y_train)
    else:
        print("Invalid model type. Choose 'nn' for neural network, 'gb' for gradient boosting, or 'svm' for support vector machine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to classify arm positions.')
    parser.add_argument('model_type', choices=['nn', 'gb', 'svm'], help="Type of model to train: 'nn' for neural network, 'gb' for gradient boosting, 'svm' for support vector machine")
    args = parser.parse_args()
    main(args.model_type)