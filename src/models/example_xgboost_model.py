from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

class XGBoostClassifier:
    def __init__(self, objective='binary:logistic', random_state=42):
        self.objective = objective
        self.random_state = random_state
        self.model = None

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        self.model = xgb.XGBClassifier(objective=self.objective, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy*100:.2f}%')
        report = classification_report(y_test, y_pred)
        print(f'Classification Report:\n{report}')

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained. Please call 'train' method first.")
        return self.model.predict(X)


