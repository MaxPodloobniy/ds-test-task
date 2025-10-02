from tensorflow.keras.datasets import mnist
from models.RandomForestClassifierMnist import RandomForestClassifierMnist
from models.NeuralNetworkClassifierMnist import NeuralNetworkClassifierMnist
from models.CNNClassifierMnist import CNNClassifierMnist

class MnistClassifier:
    def __init__(self, alg_type='cnn'):
        # Load MNIST dataset and normalize pixel values to [0,1] range
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

        # Store dataset for training and evaluation
        self.dataset = (x_train, y_train), (x_test, y_test)

        # Select the classification model based on the specified algorithm type
        if alg_type == 'cnn':
            self.model = CNNClassifierMnist()
        elif alg_type == 'nn':
            self.model = NeuralNetworkClassifierMnist()
        elif alg_type == 'rf':
            self.model = RandomForestClassifierMnist()
        else:
            raise ValueError("Invalid algorithm. Choose from 'rf', 'nn', or 'cnn'.")

    def train(self):
        # Train the selected model using the dataset
        self.model.train(self.dataset)

    def predict(self):
        # Use the trained model to make predictions on the test set
        (_, _), (X_test, _) = self.dataset
        return self.model.predict(X_test)