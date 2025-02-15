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
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf  # Добавляем импорт TensorFlow

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data[['wrist_x', 'wrist_y', 'wrist_z', 'elbow_x', 'elbow_y', 'elbow_z', 'shoulder_x', 'shoulder_y', 'shoulder_z']]
    y = data['label']
    return X, y

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Точность модели')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend(['Тренировочная', 'Валидационная'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Потери модели')
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.legend(['Тренировочная', 'Валидационная'], loc='upper left')
    
    plt.show()

def plot_feature_distributions(X):
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(X.columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(X[column], kde=True)
        plt.title(f'Распределение {column}')
    plt.tight_layout()
    plt.show()

def plot_overfitting_diagnostics(history):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Диагностика переобучения (Точность)')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend(['Тренировочная', 'Валидационная'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Диагностика переобучения (Потери)')
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.legend(['Тренировочная', 'Валидационная'], loc='upper left')
    
    plt.show()

def train_neural_network(X_train, y_train, X_val, y_val):
    input_shape = (X_train.shape[1],)
    num_classes = 1
    model = NeuralNetwork(input_shape, num_classes)
    model.build_model()
    history = model.train(X_train, y_train, X_val, y_val)
    model.save('./models/model_nn.keras')
    if history is not None:
        plot_training_history(history)
        plot_overfitting_diagnostics(history)
    else:
        print("История обучения не была возвращена.")

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, './models/model_gb.pkl')

def train_svm(X_train, y_train):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, './models/model_svm.pkl')

def main(model_type):
    # Load the dataset
    X, y = load_data('./data/dataset.csv')
    
    # Plot feature distributions
    plot_feature_distributions(X)
    
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