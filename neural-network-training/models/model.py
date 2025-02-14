import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class NeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.model = None
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=self.input_shape),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    def save(self, file_path):
        self.model.save(file_path)