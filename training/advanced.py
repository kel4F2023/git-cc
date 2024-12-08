import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re
import joblib
import os

def extract_metadata_features(commit_message):
    """Extract metadata features from commit messages."""
    features = {}
    
    # Message length
    features['msg_length'] = len(commit_message)
    
    # Number of words
    features['word_count'] = len(commit_message.split())
    
    # Contains issue reference (e.g., #123)
    features['has_issue_ref'] = int(bool(re.search(r'#\d+', commit_message)))
    
    # Contains version number
    features['has_version'] = int(bool(re.search(r'v\d+\.?\d*\.?\d*', commit_message)))
    
    # Contains file extensions
    features['has_file_extension'] = int(bool(re.search(r'\.[a-zA-Z0-9]{1,4}\b', commit_message)))
    
    # Contains URLs
    features['has_url'] = int(bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', commit_message)))
    
    # Contains code snippets (text between backticks or brackets)
    features['has_code'] = int(bool(re.search(r'`.*?`|\[.*?\]', commit_message)))
    
    # Contains common commit keywords
    keywords = ['fix', 'bug', 'feature', 'implement', 'update', 'add', 'remove', 'refactor', 'test']
    for keyword in keywords:
        features[f'has_{keyword}'] = int(bool(re.search(rf'\b{keyword}\b', commit_message.lower())))
    
    return features

def create_feature_matrix(commit_messages):
    """Create a feature matrix from a list of commit messages."""
    features_list = []
    for message in commit_messages:
        features = extract_metadata_features(message)
        features_list.append(features)
    return pd.DataFrame(features_list)

def main():
    # Load dataset
    data = pd.read_csv('data/raw.csv')
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['type'])
    
    # Extract metadata features
    metadata_features = create_feature_matrix(data['commit_message'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['commit_message'],
        encoded_labels,
        test_size=0.2,
        random_state=42
    )
    
    # Create metadata features for train and test sets
    X_train_meta = create_feature_matrix(X_train)
    X_test_meta = create_feature_matrix(X_test)
    
    # Create and configure the Bag of Words vectorizer
    bow_vectorizer = CountVectorizer(
        max_features=5000,
        min_df=2,
        stop_words='english',
        strip_accents='unicode',
        lowercase=True
    )
    
    # Transform text data
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)
    
    # Combine BOW features with metadata features
    X_train_combined = np.hstack((X_train_bow.toarray(), X_train_meta))
    X_test_combined = np.hstack((X_test_bow.toarray(), X_test_meta))
    
    # Initialize and train Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    rf_classifier.fit(X_train_combined, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test_combined)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoder.classes_))
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Feature importance analysis
    feature_names = (
        bow_vectorizer.get_feature_names_out().tolist() + 
        metadata_features.columns.tolist()
    )
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_classifier.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    # Save the model and necessary components
    model_dir = '../src/git_cc/model'
    os.makedirs(model_dir, exist_ok=True)
    
    model_components = {
        'bow_vectorizer': bow_vectorizer,
        'classifier': rf_classifier,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    joblib.dump(model_components, os.path.join(model_dir, 'advanced.joblib'))
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main() 