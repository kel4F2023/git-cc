import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    # Load dataset
    data = pd.read_csv('data/raw.csv')
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['type'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['commit_message'],
        encoded_labels,
        test_size=0.2,
        random_state=42
    )
    
    # Create and fit TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        stop_words='english',
        strip_accents='unicode',
        lowercase=True
    )
    
    # Transform the text data
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Initialize and train the Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = nb_classifier.predict(X_test_tfidf)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoder.classes_))
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Save the model and necessary components
    model_dir = '../src/git_cc/model'
    os.makedirs(model_dir, exist_ok=True)
    
    model_components = {
        'vectorizer': tfidf,
        'classifier': nb_classifier,
        'label_encoder': label_encoder
    }
    
    joblib.dump(model_components, os.path.join(model_dir, 'mini.joblib'))
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main() 