import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

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
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return y_pred, accuracy, precision, recall, f1

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Предсказано')
    plt.ylabel('Фактически')
    plt.title('Матрица ошибок')
    plt.show()

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC кривая (площадь = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложноположительных')
    plt.ylabel('Доля истинноположительных')
    plt.title('Кривая ROC')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model accuracy.')
    parser.add_argument('model_type', choices=['nn', 'gb', 'svm'], help="Type of model to evaluate: 'nn' for neural network, 'gb' for gradient boosting, 'svm' for support vector machine")
    args = parser.parse_args()

    model_path = f'./models/model_{args.model_type}.h5' if args.model_type == 'nn' else f'./models/model_{args.model_type}.pkl'
    test_data_path = './data/dataset.csv'
    
    model = load_model(args.model_type)
    X_test, y_test = load_data(test_data_path)
    y_pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, args.model_type)
    
    print(f'Точность модели: {accuracy * 100:.2f}%')
    print(f'Точность (precision): {precision * 100:.2f}%')
    print(f'Полнота (recall): {recall * 100:.2f}%')
    print(f'F1-score: {f1 * 100:.2f}%')
    
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred)