import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('data/raw.csv')
X = data['commit_message']
y = data['type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Validate the result
y_pred = model.predict(X_test)
print('Accuracy score: ', accuracy_score(y_pred, y_test))

# Save the trained model
joblib.dump(model, '../src/git_cc/model/base_classifier.joblib')