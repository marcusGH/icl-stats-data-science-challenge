import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'X' is your feature matrix and 'y' is your target variable
# X should be a DataFrame and y should be a DataFrame with binary encoding for labels

# Split the data into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Train the model on the training data
xgb_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Generate a detailed classification report
report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')
