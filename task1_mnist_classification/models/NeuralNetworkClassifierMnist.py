from tensorflow import keras
from tensorflow.keras import layers
from MnistClassifierInterface import MnistClassifierInterface


class NeuralNetworkClassifierMnist(MnistClassifierInterface):
    def __init__(self):
        self.model = keras.Sequential([
            layers.Input(shape=(28 * 28,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax")
        ])

        self.model.compile(optimizer="adam",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, dataset):
        print('\nTraining Feed-Forward Classifier...')

        (X_train, y_train), (X_test, y_test) = dataset

        X_train = X_train.reshape(X_train.shape[0], 28 * 28)  # Перетворюємо у вектор 784
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)

        self.model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))

        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f'Accuracy of model on test dataset is {test_acc}')

    def predict(self, X):
        X = X.reshape(X.shape[0], 28 * 28)

        return self.model.predict(X)

