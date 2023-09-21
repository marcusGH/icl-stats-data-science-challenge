import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import hamming_loss, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

file_name = "train-preprocessed.csv"
# Read the csv file into a dataframe
df = pd.read_csv(file_name, sep=r'\s+', quotechar='"')

# Drop columns for artist, track and album
columns_to_drop = ['Artist', 'Track', 'Album', 'Index']
df = df.drop(columns=columns_to_drop)

# Separate features from target variable
target_columns = ['A', 'B', 'C', 'D', 'AnB', 'AnC', 'AnD', 'BnC', 'BnD', 'CnD', 'AnBnC', 'AnBnD', 'AnCnD', 'BnCnD',
                  'AnBnCnD']
features_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                    'tempo', 'length', 'time_signature']
target_intersection = ['AnB', 'AnC', 'AnD', 'BnC', 'BnD', 'CnD', 'AnBnC', 'AnBnD', 'AnCnD', 'BnCnD', 'AnBnCnD']

X = df.drop(columns=target_columns)
X = X.drop(columns=['time_signature', 'length'])
y = df.drop(columns=features_columns)
y = df.drop(columns=target_intersection)

# Features and labels
features = X
labels = y

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a separate model for each label (One vs. Rest)
models = {}
for label in labels.columns:
    model = XGBClassifier(objective='binary:logistic', random_state=42)

    # Hyperparameter tuning for each model
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_micro', n_jobs=-1)
    grid_search.fit(X_train, y_train[label])

    best_model = grid_search.best_estimator_
    models[label] = best_model

# Predict probabilities for each label
y_pred_prob = pd.DataFrame()
for label, model in models.items():
    y_pred_prob[label] = model.predict_proba(X_test)[:, 1]

# Set a threshold (e.g., 0.5) for predictions
threshold = 0.5
y_pred = y_pred_prob.apply(lambda x: x >= threshold, axis=1)

# Evaluate the model
hamming_loss_score = hamming_loss(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Hamming Loss:", hamming_loss_score)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))