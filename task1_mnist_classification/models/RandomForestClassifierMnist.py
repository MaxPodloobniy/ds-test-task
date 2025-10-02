from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from MnistClassifierInterface import MnistClassifierInterface


class RandomForestClassifierMnist(MnistClassifierInterface):
    def __init__(self, n_trees=100):
        self.model = RandomForestClassifier(n_estimators=n_trees, random_state=42)

    def train(self, dataset):
        print('\nTraining Random Forest Classifier...')

        (X_train, y_train), (X_test, y_test) = dataset

        X_train = X_train.reshape(X_train.shape[0], 28 * 28)  # Перетворюємо у вектор 784
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f'Accuracy of model on test dataset is {accuracy}')

    def predict(self, X):
        X = X.reshape(X.shape[0], 28 * 28)

        return self.model.predict(X)