import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from git_cc.transformers import CommitTypeEncoder
import joblib
import os

def main():
    # Load dataset
    data = pd.read_csv('data/raw.csv')
    
    # Create and fit the label encoder
    label_encoder = CommitTypeEncoder()
    y = label_encoder.fit_transform(data['type'])
    
    # Create the feature pipeline
    feature_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            min_df=2,
            stop_words='english',
            strip_accents='unicode',
            lowercase=True
        )),
        ('classifier', MultinomialNB())
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['commit_message'],
        y,
        test_size=0.2,
        random_state=42
    )
    
    # Train the pipeline
    feature_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = feature_pipeline.predict(X_test)
    
    # Calculate and print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_encoder.commit_types_))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Save the trained pipeline and encoder separately
    model_dir = '../src/git_cc/model'
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'pipeline': feature_pipeline,
        'label_encoder': label_encoder
    }
    
    joblib.dump(model_data, os.path.join(model_dir, 'mini.joblib'))
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main() 