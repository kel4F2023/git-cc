import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from git_cc.transformers import MetadataExtractor, CommitTypeEncoder
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
        ('features', FeatureUnion([
            ('bow', CountVectorizer(
                max_features=5000,
                min_df=2,
                stop_words='english',
                strip_accents='unicode',
                lowercase=True
            )),
            ('metadata', MetadataExtractor())
        ])),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
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
    
    joblib.dump(model_data, os.path.join(model_dir, 'advanced.joblib'))
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main() 