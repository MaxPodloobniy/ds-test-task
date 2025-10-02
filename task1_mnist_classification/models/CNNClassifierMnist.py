from tensorflow import keras
from tensorflow.keras import layers
from MnistClassifierInterface import MnistClassifierInterface


class CNNClassifierMnist(MnistClassifierInterface):
    def __init__(self):
        self.model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        self.model.compile(optimizer="adam",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, dataset):
        print('\nTraining Convolutional NN Classifier...')

        (X_train, y_train), (X_test, y_test) = dataset

        self.model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f'Accuracy of model on test dataset is {test_acc}')

    def predict(self, X):
        return self.model.predict(X)

